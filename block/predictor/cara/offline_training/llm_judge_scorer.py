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

    def _generate_default_prompt(self) -> str:
        """Generate default prompt template based on scale and rationale settings."""
        # Build scale description section
        scale_section = "\n".join([
            f"{score}: {desc}"
            for score, desc in sorted(self.scale_descriptions.items())
        ])

        if self.use_rationale:
            # HF cookbook style with rationale (better performance)
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
Evaluation: (your rationale for the rating, as text)
Total rating: (your rating, as a number between {self.score_min} and {self.score_max})

You MUST provide values for 'Evaluation:' and 'Total rating:' in your answer.

Prompt: {{prompt}}

Response: {{response}}

Feedback:::
Evaluation: """
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
        with torch.no_grad():
            outputs = self._judge_model.generate(
                **inputs,
                max_new_tokens=10,
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

            # Parse rating (expect single number in configured range)
            try:
                numbers = re.findall(r'\d+(?:\.\d+)?', rating_text)
                if numbers:
                    rating = float(numbers[0])
                    # Clamp to valid range
                    clamped_rating = max(self.score_min, min(self.score_max, rating))
                    # Normalize to 0-1 range
                    range_size = self.score_max - self.score_min
                    normalized_score = (clamped_rating - self.score_min) / range_size if range_size > 0 else 0.0
                    scores[model_name] = normalized_score
                    logger.debug(
                        f"{model_name}: rating={rating}/{self.score_max} "
                        f"(normalized={normalized_score:.2f})"
                    )
                else:
                    raise ValueError(f"No number found in: {rating_text}")
            except (ValueError, IndexError) as e:
                logger.warning(
                    f"Failed to parse rating for {model_name}: '{rating_text}'. "
                    f"Error: {e}. Marking as invalid (None)"
                )
                scores[model_name] = None  # Mark as invalid, to be filtered later

        return scores
