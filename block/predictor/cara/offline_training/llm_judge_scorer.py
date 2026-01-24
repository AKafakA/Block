"""
LLM-as-a-judge model scorer.

Uses a separate LLM to evaluate response quality.
"""

import logging
from typing import List, Dict, Tuple, Optional
import torch

from block.predictor.cara.offline_training.model_scorer import ModelScorer

logger = logging.getLogger(__name__)


class LLMJudgeScorer(ModelScorer):
    """Score using an LLM as a judge.

    Uses a separate LLM to evaluate response quality based on:
    - Correctness
    - Helpfulness
    - Harmlessness
    - Coherence
    """

    # Default 10-point scale with detailed guidance
    DEFAULT_SCALE_DESCRIPTIONS = {
        1: "Completely incorrect, irrelevant, or unhelpful",
        2: "Mostly incorrect or missing critical information",
        3: "Partially correct but with significant issues",
        4: "Somewhat helpful but with notable gaps",
        5: "Moderately helpful, addresses some key points",
        6: "Generally helpful with minor issues",
        7: "Good response, addresses most points well",
        8: "Very good, comprehensive and accurate",
        9: "Excellent, thorough and well-structured",
        10: "Perfect, exemplary response in all aspects"
    }

    def __init__(self,
                 judge_model: str = "Unbabel/M-Prometheus-7B",
                 judge_prompt_template: Optional[str] = None,
                 batch_size: int = 1,
                 device: str = "auto",
                 hf_token: Optional[str] = None,
                 score_min: int = 1,
                 score_max: int = 10,
                 scale_descriptions: Optional[Dict[int, str]] = None,
                 use_rationale: bool = True,
                 ):
        """
        Args:
            judge_model: HuggingFace model name or local path for judge LLM
            judge_prompt_template: Custom prompt template for judging.
                                  Must contain {prompt} and {response} placeholders.
            batch_size: Batch size for judge model inference
            device: Device for judge model ("auto", "cuda", "cpu")
            hf_token: HuggingFace API token for gated models
            score_min: Minimum score value (default: 1)
            score_max: Maximum score value (default: 10)
            scale_descriptions: Optional dict mapping scores to descriptions.
                               If None, auto-generates or uses defaults for 1-10 scale.
            use_rationale: If True, prompts LLM to provide reasoning before rating.
                          Improves accuracy based on HF cookbook findings.
        """
        self.judge_model_name = judge_model
        self.batch_size = batch_size
        self.device = device
        self.hf_token = hf_token
        self.score_min = score_min
        self.score_max = score_max
        self.use_rationale = use_rationale

        # Setup scale descriptions
        if scale_descriptions:
            self.scale_descriptions = scale_descriptions
        elif score_min == 1 and score_max == 10:
            self.scale_descriptions = self.DEFAULT_SCALE_DESCRIPTIONS
        else:
            # Auto-generate simple descriptions for custom scales
            self.scale_descriptions = self._generate_scale_descriptions(score_min, score_max)

        # Generate or use provided template
        if judge_prompt_template:
            self.judge_prompt_template = judge_prompt_template
        else:
            self.judge_prompt_template = self._generate_default_prompt()

        # Validate template
        if "{prompt}" not in self.judge_prompt_template or \
           "{response}" not in self.judge_prompt_template:
            raise ValueError(
                "judge_prompt_template must contain {prompt} and {response} placeholders"
            )

        # Lazy load judge model
        self._judge_model = None
        self._judge_tokenizer = None

        # Statistics tracking for parsing methods
        self._stats = {
            'total_attempts': 0,
            'number_extraction': 0,
            'exact_match': 0,
            'semantic_match': 0,
            'failures': 0,
            'semantic_similarities': [],  # Track similarity scores
        }

        logger.info(
            f"LLMJudgeScorer initialized: model={judge_model}, "
            f"scale={score_min}-{score_max}, use_rationale={use_rationale}, "
            f"batch_size={batch_size}, device={device}"
        )

    def _generate_scale_descriptions(self, min_score: int, max_score: int) -> Dict[int, str]:
        """Auto-generate scale descriptions for custom ranges."""
        range_size = max_score - min_score + 1
        descriptions = {}

        if range_size == 4:
            # Use HF cookbook 1-4 style
            descriptions[min_score] = "Terrible: completely irrelevant or very partial"
            descriptions[min_score + 1] = "Mostly not helpful: misses key aspects"
            descriptions[min_score + 2] = "Mostly helpful: provides support but could be improved"
            descriptions[max_score] = "Excellent: relevant, direct, and fully addresses the prompt"
        elif range_size == 5:
            # 5-point scale (common in surveys)
            descriptions[min_score] = "Very poor"
            descriptions[min_score + 1] = "Poor"
            descriptions[min_score + 2] = "Fair"
            descriptions[min_score + 3] = "Good"
            descriptions[max_score] = "Excellent"
        else:
            # Generic descriptions
            for i, score in enumerate(range(min_score, max_score + 1)):
                if i == 0:
                    descriptions[score] = "Completely inadequate"
                elif i == range_size - 1:
                    descriptions[score] = "Perfect, exemplary"
                else:
                    # Interpolate quality
                    quality_pct = i / (range_size - 1)
                    if quality_pct < 0.33:
                        descriptions[score] = "Below expectations"
                    elif quality_pct < 0.67:
                        descriptions[score] = "Meets some expectations"
                    else:
                        descriptions[score] = "Above expectations"

        return descriptions

    def _match_score_exact(self, text: str) -> Optional[float]:
        """Stage 2: Try to find exact substring match against scale descriptions.

        Args:
            text: The generated text

        Returns:
            Score if exact match found, None otherwise
        """
        text_lower = text.lower().strip()

        # Try to find exact substring matches
        for score, description in self.scale_descriptions.items():
            desc_lower = description.lower().strip()

            # Check if description appears as substring in text
            if desc_lower in text_lower or text_lower in desc_lower:
                logger.debug(
                    f"Exact substring match: '{text[:60]}...' matched '{desc_lower}' -> score {score}"
                )
                return float(score)

        # No exact match found
        return None

    def _match_score_by_embedding(self, text: str) -> Optional[Tuple[float, float]]:
        """Stage 3: Match text to score using semantic similarity with embeddings.

        If the model outputs description text instead of a number,
        use embeddings to find the most similar scale description.

        Args:
            text: The generated text

        Returns:
            Tuple of (score, similarity) if a good match is found, None otherwise
        """
        try:
            from sentence_transformers import SentenceTransformer, util
            import torch

            # Lazy load embedding model
            if not hasattr(self, '_embedding_model'):
                logger.debug("Loading sentence embedding model for fallback parsing...")
                self._embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight & fast
                logger.debug("Embedding model loaded")

            # Get embeddings for the generated text
            text_embedding = self._embedding_model.encode(text, convert_to_tensor=True)

            # Get embeddings for all scale descriptions
            descriptions = list(self.scale_descriptions.values())
            scores = list(self.scale_descriptions.keys())

            desc_embeddings = self._embedding_model.encode(descriptions, convert_to_tensor=True)

            # Compute cosine similarities
            similarities = util.cos_sim(text_embedding, desc_embeddings)[0]

            # Find best match
            best_idx = similarities.argmax().item()
            best_score = scores[best_idx]
            best_similarity = similarities[best_idx].item()

            # Only return if similarity is high enough (> 0.5 threshold)
            if best_similarity > 0.5:
                logger.debug(
                    f"Semantic match: '{text[:60]}...' -> score {best_score} "
                    f"(similarity: {best_similarity:.3f}, matched: '{descriptions[best_idx]}')"
                )
                return (float(best_score), float(best_similarity))
            else:
                logger.debug(
                    f"Semantic similarity too low ({best_similarity:.3f}), rejecting match"
                )
                return None

        except ImportError:
            logger.warning(
                "sentence-transformers not available for semantic fallback parsing. "
                "Install with: pip install sentence-transformers"
            )
            return None
        except Exception as e:
            logger.warning(f"Semantic matching failed: {e}")
            return None

    def _generate_default_prompt(self) -> str:
        """Generate default prompt template based on scale and rationale settings."""
        # Build scale description section
        scale_section = "\n".join([
            f"{score}: {desc}"
            for score, desc in sorted(self.scale_descriptions.items())
        ])

        if self.use_rationale:
            # HF cookbook style with rationale (better performance)
            # Put rating FIRST to ensure it's generated within token limit
            return f"""You are a helpful assistant evaluating the quality of AI responses.

Given the following prompt and response, rate the response quality on a scale of {self.score_min} to {self.score_max}.

Here is the scale you should use:
{scale_section}

Consider these aspects:
- Correctness: Is the response factually accurate?
- Helpfulness: Does it address the prompt effectively?
- Coherence: Is it well-written and clear?

Provide your feedback as follows:

Feedback:::
Total rating: (your rating, as a number between {self.score_min} and {self.score_max})
Evaluation: (your rationale for the rating, as text)

You MUST provide a number for 'Total rating:' first, then your reasoning in 'Evaluation:'.

Prompt: {{prompt}}

Response: {{response}}

Feedback:::
Total rating: """
        else:
            # Simple style (faster but less accurate)
            return f"""You are a helpful assistant evaluating the quality of AI responses.

Given the following prompt and response, rate the response quality on a scale of {self.score_min} to {self.score_max}.

Scale:
{scale_section}

Consider: correctness, helpfulness, and coherence.

Prompt: {{prompt}}

Response: {{response}}

Provide ONLY a single number between {self.score_min} and {self.score_max} as your rating.
Rating:"""

    def _load_judge_model(self):
        """Lazy load judge model and tokenizer."""
        if self._judge_model is not None:
            return

        logger.info(f"Loading judge model: {self.judge_model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Load tokenizer, forwarding HF token if provided.
            try:
                if self.hf_token:
                    self._judge_tokenizer = AutoTokenizer.from_pretrained(
                        self.judge_model_name,
                        trust_remote_code=True,
                        padding_side="left",
                        token=self.hf_token,  # Transformers >= 4.46 / v5
                    )
                else:
                    self._judge_tokenizer = AutoTokenizer.from_pretrained(
                        self.judge_model_name,
                        trust_remote_code=True,
                        padding_side="left",
                    )
            except TypeError:
                # Backwards compatibility with older Transformers
                if self.hf_token:
                    self._judge_tokenizer = AutoTokenizer.from_pretrained(
                        self.judge_model_name,
                        trust_remote_code=True,
                        padding_side="left",
                        use_auth_token=self.hf_token,  # Older API
                    )
                else:
                    raise
            # Set pad token to suppress warning during generation
            if self._judge_tokenizer.pad_token is None:
                self._judge_tokenizer.pad_token = self._judge_tokenizer.eos_token

            # Load model with appropriate dtype, forwarding HF token if provided.
            try:
                if self.hf_token:
                    self._judge_model = AutoModelForCausalLM.from_pretrained(
                        self.judge_model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                        token=self.hf_token,
                    )
                else:
                    self._judge_model = AutoModelForCausalLM.from_pretrained(
                        self.judge_model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                    )
            except TypeError:
                if self.hf_token:
                    self._judge_model = AutoModelForCausalLM.from_pretrained(
                        self.judge_model_name,
                        trust_remote_code=True,
                        torch_dtype=torch.float16,
                        device_map=self.device,
                        use_auth_token=self.hf_token,
                    )
                else:
                    raise

            self._judge_model.eval()

            logger.info(f"Judge model loaded successfully on {self._judge_model.device}")

        except Exception as e:
            logger.error(f"Failed to load judge model: {e}")
            raise

    def get_parsing_stats(self) -> Dict:
        """Get statistics about parsing methods used.

        Returns:
            Dict with counts and metrics for each parsing method
        """
        stats = self._stats.copy()

        # Compute averages
        if stats['semantic_similarities']:
            stats['avg_semantic_similarity'] = sum(stats['semantic_similarities']) / len(stats['semantic_similarities'])
            stats['min_semantic_similarity'] = min(stats['semantic_similarities'])
            stats['max_semantic_similarity'] = max(stats['semantic_similarities'])
        else:
            stats['avg_semantic_similarity'] = None
            stats['min_semantic_similarity'] = None
            stats['max_semantic_similarity'] = None

        # Compute success rates
        if stats['total_attempts'] > 0:
            stats['success_rate'] = (stats['total_attempts'] - stats['failures']) / stats['total_attempts']
            stats['number_extraction_rate'] = stats['number_extraction'] / stats['total_attempts']
            stats['exact_match_rate'] = stats['exact_match'] / stats['total_attempts']
            stats['semantic_match_rate'] = stats['semantic_match'] / stats['total_attempts']
            stats['failure_rate'] = stats['failures'] / stats['total_attempts']
        else:
            stats['success_rate'] = 0.0
            stats['number_extraction_rate'] = 0.0
            stats['exact_match_rate'] = 0.0
            stats['semantic_match_rate'] = 0.0
            stats['failure_rate'] = 0.0

        # Remove raw similarities list from output (too long)
        del stats['semantic_similarities']

        return stats

    def print_parsing_stats(self):
        """Print formatted parsing statistics."""
        stats = self.get_parsing_stats()

        print(f"\n{'='*60}")
        print(f"PARSING STATISTICS - {self.judge_model_name}")
        print(f"{'='*60}")
        print(f"Total attempts:       {stats['total_attempts']}")
        print(f"Overall success rate: {stats['success_rate']*100:.1f}%")
        print(f"\nParsing Method Breakdown:")
        print(f"  Stage 1 (Number extraction): {stats['number_extraction']:>5} ({stats['number_extraction_rate']*100:>5.1f}%)")
        print(f"  Stage 2 (Exact substring):   {stats['exact_match']:>5} ({stats['exact_match_rate']*100:>5.1f}%)")
        print(f"  Stage 3 (Semantic embedding):{stats['semantic_match']:>5} ({stats['semantic_match_rate']*100:>5.1f}%)")
        print(f"  Failed:                       {stats['failures']:>5} ({stats['failure_rate']*100:>5.1f}%)")

        if stats['avg_semantic_similarity'] is not None:
            print(f"\nSemantic Matching Confidence:")
            print(f"  Average similarity: {stats['avg_semantic_similarity']:.3f}")
            print(f"  Range: [{stats['min_semantic_similarity']:.3f}, {stats['max_semantic_similarity']:.3f}]")
            print(f"  (Higher is better, threshold is 0.5)")

        print(f"{'='*60}\n")

    def score(self,
              prompt: str,
              responses: List[Tuple[str, str]]) -> Dict[str, float]:
        """Compute LLM-judge quality scores.

        Args:
            prompt: Input prompt
            responses: List of (model_name, generated_text) tuples

        Returns:
            Dict mapping model_name -> quality_score (0.0-1.0)
        """
        if not responses:
            return {}

        # Load judge model if needed
        self._load_judge_model()

        scores = {}

        # Process in batches
        for i in range(0, len(responses), self.batch_size):
            batch = responses[i:i + self.batch_size]
            batch_scores = self._score_batch(prompt, batch)
            scores.update(batch_scores)

        return scores

    def score_pairs(self,
                    pairs: List[Tuple[str, str, str]]) -> List[Optional[float]]:
        """Score multiple (prompt, model_name, response) pairs in batches.

        This enables batching across different requests (different prompts).

        Args:
            pairs: List of tuples (prompt, model_name, response)

        Returns:
            List of normalized scores (0.0-1.0) or None for failures,
            aligned with the input order.
        """
        import re
        import torch

        if not pairs:
            return []

        self._load_judge_model()

        results: List[Optional[float]] = [None] * len(pairs)

        # Process in chunks according to batch size
        for start in range(0, len(pairs), self.batch_size):
            chunk = pairs[start:start + self.batch_size]

            # Format prompts and keep model names for logging
            model_names: List[str] = []
            judge_prompts: List[str] = []
            for prompt, model_name, generated_text in chunk:
                judge_prompts.append(self.judge_prompt_template.format(
                    prompt=prompt,
                    response=generated_text
                ))
                model_names.append(model_name)

            # Tokenize with padding
            inputs = self._judge_tokenizer(
                judge_prompts,
                return_tensors="pt",
                truncation=True,
                max_length=2048,
                padding=True,
            ).to(self._judge_model.device)

            max_tokens = 200 if self.use_rationale else 10
            with torch.no_grad():
                outputs = self._judge_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    do_sample=False,
                    pad_token_id=self._judge_tokenizer.pad_token_id,
                )

            # Parse outputs: generated tokens start after the padded input length
            common_input_len = inputs['input_ids'].shape[1]
            for local_idx, model_name in enumerate(model_names):
                global_idx = start + local_idx
                try:
                    generated_tokens = outputs[local_idx][common_input_len:]
                    rating_text = self._judge_tokenizer.decode(
                        generated_tokens,
                        skip_special_tokens=True
                    ).strip()

                    self._stats['total_attempts'] += 1

                    rating = None
                    parse_method = None

                    # Stage 1: numeric extraction
                    rating_section = rating_text
                    if "Evaluation:" in rating_text:
                        rating_section = rating_text.split("Evaluation:")[0]
                    numbers = re.findall(r'\d+(?:\.\d+)?', rating_section)
                    if numbers:
                        rating = float(numbers[0])
                        parse_method = 'number_extraction'
                        self._stats['number_extraction'] += 1
                        logger.debug(
                            f"{model_name}: [STAGE 1] Number extraction: {rating}/{self.score_max}"
                        )

                    # Stage 2: exact substring match
                    if rating is None:
                        logger.debug(
                            f"{model_name}: [STAGE 2] No number found, trying exact substring match..."
                        )
                        matched_score = self._match_score_exact(rating_text)
                        if matched_score is not None:
                            rating = matched_score
                            parse_method = 'exact_match'
                            self._stats['exact_match'] += 1

                    # Stage 3: semantic match
                    if rating is None:
                        logger.debug(
                            f"{model_name}: [STAGE 3] No exact match, trying semantic embedding match..."
                        )
                        match_result = self._match_score_by_embedding(rating_text)
                        if match_result is not None:
                            matched_score, similarity = match_result
                            rating = matched_score
                            parse_method = 'semantic_match'
                            self._stats['semantic_match'] += 1
                            self._stats['semantic_similarities'].append(similarity)
                            logger.debug(
                                f"{model_name}: [STAGE 3] Semantic match: {rating}/{self.score_max} "
                                f"(similarity: {similarity:.3f})"
                            )

                    if rating is not None:
                        clamped_rating = max(self.score_min, min(self.score_max, rating))
                        range_size = self.score_max - self.score_min
                        normalized_score = (clamped_rating - self.score_min) / range_size if range_size > 0 else 0.0
                        results[global_idx] = normalized_score
                        logger.debug(
                            f"{model_name}: ✓ SUCCESS via {parse_method}: rating={rating}/{self.score_max} "
                            f"(normalized={normalized_score:.3f})"
                        )
                    else:
                        self._stats['failures'] += 1
                        results[global_idx] = None
                except Exception as e:
                    logger.warning(
                        f"Failed to parse rating for {model_name}: error: {e}. Marking as invalid (None)"
                    )
                    self._stats['failures'] += 1
                    results[global_idx] = None

        return results
    def _score_batch(self,
                     prompt: str,
                     batch: List[Tuple[str, str]]) -> Dict[str, float]:
        """Score a batch of responses with true batched inference.

        Args:
            prompt: Input prompt
            batch: Batch of (model_name, generated_text) tuples

        Returns:
            Dict mapping model_name -> quality_score (0.0-1.0)
        """
        import re

        if not batch:
            return {}

        scores = {}
        model_names = []
        judge_prompts = []

        # Format all judge prompts
        for model_name, generated_text in batch:
            judge_prompt = self.judge_prompt_template.format(
                prompt=prompt,
                response=generated_text
            )
            model_names.append(model_name)
            judge_prompts.append(judge_prompt)

        # Tokenize batch with padding
        inputs = self._judge_tokenizer(
            judge_prompts,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
            padding=True
        ).to(self._judge_model.device)

        # Generate ratings for entire batch
        # Use more tokens if rationale is enabled
        max_tokens = 200 if self.use_rationale else 10

        with torch.no_grad():
            outputs = self._judge_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=False,
                pad_token_id=self._judge_tokenizer.pad_token_id
            )

        # Decode and parse each output
        for idx, model_name in enumerate(model_names):
            # Extract only the generated tokens (after input)
            generated_tokens = outputs[idx][inputs['input_ids'].shape[1]:]
            rating_text = self._judge_tokenizer.decode(
                generated_tokens,
                skip_special_tokens=True
            ).strip()

            # Parse rating with three-stage approach:
            # Stage 1: Number extraction (direct)
            # Stage 2: Exact substring match (against scale descriptions)
            # Stage 3: Semantic embedding match (last resort)
            self._stats['total_attempts'] += 1

            try:
                # Prepare rating section - look for number right after the prompt end
                # If there's an "Evaluation:" marker, extract only the part before it
                # to avoid picking up numbers from the evaluation text
                rating_section = rating_text
                if "Evaluation:" in rating_text:
                    rating_section = rating_text.split("Evaluation:")[0]

                rating = None
                parse_method = None

                # STAGE 1: Try to extract number directly
                numbers = re.findall(r'\d+(?:\.\d+)?', rating_section)
                if numbers:
                    rating = float(numbers[0])
                    parse_method = 'number_extraction'
                    self._stats['number_extraction'] += 1
                    logger.debug(
                        f"{model_name}: [STAGE 1] Number extraction: {rating}/{self.score_max}"
                    )

                # STAGE 2: Try exact substring match
                if rating is None:
                    logger.debug(
                        f"{model_name}: [STAGE 2] No number found, trying exact substring match..."
                    )
                    matched_score = self._match_score_exact(rating_text)
                    if matched_score is not None:
                        rating = matched_score
                        parse_method = 'exact_match'
                        self._stats['exact_match'] += 1

                # STAGE 3: Try semantic embedding match
                if rating is None:
                    logger.debug(
                        f"{model_name}: [STAGE 3] No exact match, trying semantic embedding match..."
                    )
                    match_result = self._match_score_by_embedding(rating_text)
                    if match_result is not None:
                        matched_score, similarity = match_result
                        rating = matched_score
                        parse_method = 'semantic_match'
                        self._stats['semantic_match'] += 1
                        self._stats['semantic_similarities'].append(similarity)
                        logger.debug(
                            f"{model_name}: [STAGE 3] Semantic match: {rating}/{self.score_max} "
                            f"(similarity: {similarity:.3f})"
                        )

                # Check if we got a rating from any stage
                if rating is not None:
                    # Clamp to valid range
                    clamped_rating = max(self.score_min, min(self.score_max, rating))
                    # Normalize to 0-1 range
                    range_size = self.score_max - self.score_min
                    normalized_score = (clamped_rating - self.score_min) / range_size if range_size > 0 else 0.0
                    scores[model_name] = normalized_score
                    logger.debug(
                        f"{model_name}: ✓ SUCCESS via {parse_method}: rating={rating}/{self.score_max} "
                        f"(normalized={normalized_score:.3f})"
                    )
                else:
                    # All stages failed
                    self._stats['failures'] += 1
                    raise ValueError(
                        f"All parsing stages failed for text: {rating_text[:150]}..."
                    )

            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Failed to parse rating for {model_name}: '{rating_text[:200]}...'. "
                    f"Error: {e}. Marking as invalid (None)"
                )
                scores[model_name] = None  # Mark as invalid, to be filtered later

        return scores
