"""
Similarity-based model scorer using semantic embeddings.

Scores model responses based on embedding similarity to a reference model's response.
Uses sentence transformers for semantic similarity (like GPTCache).
"""

import logging
import re
from typing import List, Dict, Tuple, Optional
import torch
import torch.nn.functional as F

from block.predictor.cara.offline_training.model_scorer import ModelScorer

logger = logging.getLogger(__name__)


class SimilarityScorer(ModelScorer):
    """Score based on embedding similarity to reference model's response.

    Reference model defaults to largest model (by parameter count).
    Uses sentence-transformers to compute semantic similarity.
    """

    def __init__(self,
                 reference_model: Optional[str] = None,
                 embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2",
                 device: str = "cpu"):
        """
        Args:
            reference_model: Model name to use as reference.
                           If None, automatically selects largest model.
            embedding_model: Sentence transformer model for embeddings.
                           Default: all-MiniLM-L6-v2 (fast, 384-dim)
                           Other options:
                           - all-mpnet-base-v2 (slower, 768-dim, better quality)
                           - all-MiniLM-L12-v2 (medium, 384-dim)
            device: Device for embedding model ("cpu", "cuda")
        """
        self.reference_model = reference_model
        self.embedding_model_name = embedding_model
        self.device = torch.device(device)

        # Lazy load embedding model
        self._embedding_model = None

        logger.info(
            f"SimilarityScorer initialized: "
            f"reference={reference_model or 'auto (largest)'}, "
            f"embedding_model={embedding_model}, device={device}"
        )

    def _load_embedding_model(self):
        """Lazy load sentence transformer model."""
        if self._embedding_model is not None:
            return

        logger.info(f"Loading embedding model: {self.embedding_model_name}")

        try:
            from sentence_transformers import SentenceTransformer

            self._embedding_model = SentenceTransformer(
                self.embedding_model_name,
                device=str(self.device)
            )

            logger.info(
                f"Embedding model loaded: "
                f"dim={self._embedding_model.get_sentence_embedding_dimension()}"
            )

        except ImportError:
            logger.error(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )
            raise
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def score(self,
              prompt: str,
              responses: List[Tuple[str, str]]) -> Dict[str, float]:
        """Compute embedding-based similarity scores.

        Args:
            prompt: Input prompt (unused for similarity scoring)
            responses: List of (model_name, generated_text) tuples

        Returns:
            Dict mapping model_name -> similarity_score (0.0-1.0)
        """
        if not responses:
            return {}

        # Load embedding model if needed
        self._load_embedding_model()

        # Determine reference model
        reference_model = self.reference_model
        if reference_model is None:
            reference_model = self._find_largest_model([r[0] for r in responses])

        # Find reference response
        reference_text = None
        for model_name, generated_text in responses:
            if model_name == reference_model:
                reference_text = generated_text
                break

        if reference_text is None:
            logger.warning(
                f"Reference model {reference_model} not found. "
                f"Using first model: {responses[0][0]}"
            )
            reference_model, reference_text = responses[0]

        logger.debug(f"Using {reference_model} as reference")

        # Encode all responses to embeddings
        texts = [generated_text for _, generated_text in responses]

        # Get embeddings as PyTorch tensors
        embeddings = self._embedding_model.encode(
            texts,
            convert_to_tensor=True,
            device=self.device,
            show_progress_bar=False
        )

        # Find reference embedding
        reference_idx = None
        for idx, (model_name, _) in enumerate(responses):
            if model_name == reference_model:
                reference_idx = idx
                break

        reference_embedding = embeddings[reference_idx].unsqueeze(0)  # [1, dim]

        # Compute cosine similarity using PyTorch
        # F.cosine_similarity expects inputs of shape [N, dim]
        similarities = F.cosine_similarity(
            embeddings,
            reference_embedding,
            dim=1
        )

        # Build scores dict
        scores = {}
        for idx, (model_name, _) in enumerate(responses):
            # Cosine similarity is in [-1, 1], normalize to [0, 1]
            similarity = similarities[idx].item()
            normalized_similarity = (similarity + 1.0) / 2.0
            scores[model_name] = normalized_similarity

            if model_name != reference_model:
                logger.debug(
                    f"{model_name} similarity to {reference_model}: "
                    f"{normalized_similarity:.4f} (raw: {similarity:.4f})"
                )

        return scores

    def _find_largest_model(self, model_names: List[str]) -> str:
        """Find largest model by parsing parameter count from name.

        Examples:
            "Qwen/Qwen2.5-72B" -> 72.0
            "Qwen/Qwen2.5-3B" -> 3.0
            "meta-llama/Llama-3-70b-hf" -> 70.0
        """
        model_sizes = {}

        for model_name in model_names:
            # Extract size from model name (e.g., "72B", "3B", "70b")
            match = re.search(r'(\d+(?:\.\d+)?)[Bb]', model_name)
            if match:
                size = float(match.group(1))
                model_sizes[model_name] = size
            else:
                # Unknown size, assume 0
                model_sizes[model_name] = 0.0

        if not model_sizes:
            # Fallback to first model
            return model_names[0]

        largest_model = max(model_sizes.items(), key=lambda x: x[1])[0]
        logger.debug(
            f"Largest model identified: {largest_model} "
            f"({model_sizes[largest_model]:.1f}B)"
        )
        return largest_model