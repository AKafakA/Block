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

    DEFAULT_JUDGE_PROMPT = """You are a helpful assistant evaluating the quality of AI responses.

Given the following prompt and response, rate the response quality on a scale of 0-10.

Consider:
- Correctness: Is the response factually accurate?
- Helpfulness: Does it address the prompt effectively?
- Harmlessness: Is it safe and appropriate?
- Coherence: Is it well-written and clear?

Prompt: {prompt}

Response: {response}

Provide ONLY a single number between 0 and 10 as your rating.
Rating:"""

    def __init__(self,
                 judge_model: str = "Unbabel/M-Prometheus-7B",
                 judge_prompt_template: Optional[str] = None,
                 batch_size: int = 1,
                 device: str = "auto"):
        """
        Args:
            judge_model: HuggingFace model name or local path for judge LLM
            judge_prompt_template: Custom prompt template for judging.
                                  Must contain {prompt} and {response} placeholders.
            batch_size: Batch size for judge model inference
            device: Device for judge model ("auto", "cuda", "cpu")
        """
        self.judge_model_name = judge_model
        self.batch_size = batch_size
        self.device = device

        # Use provided template or default
        self.judge_prompt_template = (
            judge_prompt_template or self.DEFAULT_JUDGE_PROMPT
        )

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
            f"batch_size={batch_size}, device={device}"
        )

    def _load_judge_model(self):
        """Lazy load judge model and tokenizer."""
        if self._judge_model is not None:
            return

        logger.info(f"Loading judge model: {self.judge_model_name}")

        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # Prefer built-in fast tokenizers without remote code.
            # This avoids misconfigured tokenizer_config (e.g. TokenizersBackend).
            try:
                self._judge_tokenizer = AutoTokenizer.from_pretrained(
                    self.judge_model_name,
                    use_fast=True,
                    trust_remote_code=False,
                    padding_side="left",  # Left padding for batched generation
                )
            except Exception as te:
                logger.warning(
                    f"AutoTokenizer load without remote code failed: {te}. "
                    f"Retrying with trust_remote_code=True."
                )
                # Fallback: allow remote code in case the model truly needs it
                self._judge_tokenizer = AutoTokenizer.from_pretrained(
                    self.judge_model_name,
                    use_fast=True,
                    trust_remote_code=True,
                    padding_side="left",
                )
            # Set pad token to suppress warning during generation
            if self._judge_tokenizer.pad_token is None:
                self._judge_tokenizer.pad_token = self._judge_tokenizer.eos_token

            # Load model with appropriate dtype. Prefer built-in classes first.
            try:
                self._judge_model = AutoModelForCausalLM.from_pretrained(
                    self.judge_model_name,
                    trust_remote_code=False,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                )
            except Exception as me:
                logger.warning(
                    f"AutoModel load without remote code failed: {me}. "
                    f"Retrying with trust_remote_code=True."
                )
                self._judge_model = AutoModelForCausalLM.from_pretrained(
                    self.judge_model_name,
                    trust_remote_code=True,
                    torch_dtype=torch.float16,
                    device_map=self.device,
                )

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

            # Parse rating (expect single number 0-10)
            try:
                numbers = re.findall(r'\d+(?:\.\d+)?', rating_text)
                if numbers:
                    rating = float(numbers[0])
                    normalized_score = max(0.0, min(10.0, rating)) / 10.0
                    scores[model_name] = normalized_score
                    logger.debug(
                        f"{model_name}: rating={rating}/10 ({normalized_score:.2f})"
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
