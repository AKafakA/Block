#!/usr/bin/env python3
"""
KNN-based estimator for CARA output length and quality prediction.

Uses sentence-transformer embeddings + nearest neighbor search.
For a new prompt: embed → find top-k neighbors → aggregate per-model values.
"""

import json
import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


class KNNEstimator:
    """KNN-based length and quality estimator.

    Stores training embeddings and per-model labels (output_length, quality scores).
    At inference: embed query prompt, find k nearest neighbors, return
    distance-weighted aggregation of neighbor values.

    Supports both old schema (quality_score float) and new schema
    (similarity_score + llm_judge_scores dict). For the new schema,
    quality is computed as: 0.5 * similarity_score + 0.5 * mean(llm_judge_scores).
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        k: int = 10,
        device: str = "cpu",
    ):
        self.embedding_model_name = embedding_model_name
        self.k = k
        self.device = device

        # Loaded at build/load time
        self.embeddings: Optional[np.ndarray] = None  # (N, dim)
        self.model_names: List[str] = []
        # Per-model arrays: model_name -> (N,) array
        self.output_lengths: Dict[str, np.ndarray] = {}
        self.quality_scores: Dict[str, np.ndarray] = {}
        self.similarity_scores: Dict[str, np.ndarray] = {}
        self.llm_judge_scores: Dict[str, np.ndarray] = {}

        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
            logger.info(f"Embedding model loaded: dim={self._encoder.get_sentence_embedding_dimension()}")
        return self._encoder

    @staticmethod
    def _extract_quality(m_data: Dict) -> float:
        """Extract a combined quality score from model data.

        Supports both old schema (quality_score) and new schema
        (similarity_score + llm_judge_scores). For new schema:
        quality = 0.5 * similarity + 0.5 * mean(judge_scores).
        Falls back gracefully if only one score type is available.
        """
        # Old schema: direct quality_score
        if "quality_score" in m_data:
            return float(m_data["quality_score"])

        # New schema: combine similarity + judge scores
        sim = m_data.get("similarity_score")
        judge_scores = m_data.get("llm_judge_scores", {})
        valid_judges = [v for v in judge_scores.values() if v is not None]
        judge_mean = sum(valid_judges) / len(valid_judges) if valid_judges else None

        if sim is not None and judge_mean is not None:
            return 0.5 * float(sim) + 0.5 * float(judge_mean)
        elif sim is not None:
            return float(sim)
        elif judge_mean is not None:
            return float(judge_mean)
        return 0.0

    def build_index(self, training_data: List[Dict]):
        """Build KNN index from processed training data.

        Args:
            training_data: List of dicts with {prompt, input_len, models: {model: {output_length, ...}}}
                Supports both old schema (quality_score, ttft) and new schema
                (similarity_score, llm_judge_scores dict).
        """
        n = len(training_data)
        logger.info(f"Building KNN index from {n} training examples...")

        # Extract prompts and encode
        prompts = [d["prompt"] for d in training_data]
        encoder = self._get_encoder()
        self.embeddings = encoder.encode(
            prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        logger.info(f"Embeddings shape: {self.embeddings.shape}")

        # Collect model names from first example
        self.model_names = sorted(training_data[0]["models"].keys())

        # Detect schema
        first_model_data = next(iter(training_data[0]["models"].values()))
        has_new_schema = "similarity_score" in first_model_data or "llm_judge_scores" in first_model_data
        if has_new_schema:
            logger.info("Detected new schema (similarity_score + llm_judge_scores)")
        else:
            logger.info("Detected old schema (quality_score)")

        # Build per-model label arrays
        for model in self.model_names:
            lengths = []
            qualities = []
            sim_scores = []
            judge_scores = []
            for d in training_data:
                m_data = d["models"].get(model, {})
                lengths.append(m_data.get("output_length", 0))
                qualities.append(self._extract_quality(m_data))
                sim_scores.append(m_data.get("similarity_score", 0.0))
                # Store mean of judge scores for this sample
                js = m_data.get("llm_judge_scores", {})
                valid_js = [v for v in js.values() if v is not None]
                judge_scores.append(sum(valid_js) / len(valid_js) if valid_js else 0.0)

            self.output_lengths[model] = np.array(lengths, dtype=np.float32)
            self.quality_scores[model] = np.array(qualities, dtype=np.float32)
            self.similarity_scores[model] = np.array(sim_scores, dtype=np.float32)
            self.llm_judge_scores[model] = np.array(judge_scores, dtype=np.float32)

        logger.info(
            f"Index built: {n} examples, {len(self.model_names)} models, "
            f"embedding_dim={self.embeddings.shape[1]}"
        )

    def predict_length(
        self, prompt: str, model_name: str
    ) -> Dict[str, float]:
        """Predict output length for a prompt on a given model.

        Returns:
            Dict with mean, p50, p90, p95 predictions
        """
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances)

        values = self.output_lengths[model_name][indices]
        weighted_mean = float(np.average(values, weights=weights))

        return {
            "mean": weighted_mean,
            "p50": float(np.median(values)),
            "p90": float(np.percentile(values, 90)),
            "p95": float(np.percentile(values, 95)),
            "raw_values": values.tolist(),
        }

    def predict_quality(
        self, prompt: str, model_name: str
    ) -> float:
        """Predict quality score for a prompt on a given model."""
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances)
        values = self.quality_scores[model_name][indices]
        return float(np.average(values, weights=weights))

    def predict_all_models(
        self, prompt: str
    ) -> Dict[str, Dict[str, float]]:
        """Predict length and quality for all models at once.

        Returns:
            {model_name: {length_mean, length_p95, quality_score}}
        """
        indices, distances = self._find_neighbors(prompt)
        weights = self._distance_weights(distances)

        results = {}
        for model in self.model_names:
            lengths = self.output_lengths[model][indices]
            qualities = self.quality_scores[model][indices]
            results[model] = {
                "length_mean": float(np.average(lengths, weights=weights)),
                "length_p95": float(np.percentile(lengths, 95)),
                "quality_score": float(np.average(qualities, weights=weights)),
            }
        return results

    def _find_neighbors(self, prompt: str) -> Tuple[np.ndarray, np.ndarray]:
        """Find k nearest neighbors for a prompt.

        Returns:
            (indices, distances) arrays of shape (k,)
        """
        encoder = self._get_encoder()
        query = encoder.encode([prompt], normalize_embeddings=True)

        # Cosine similarity (embeddings are normalized, so dot product = cosine sim)
        similarities = self.embeddings @ query.T  # (N, 1)
        similarities = similarities.squeeze()  # (N,)

        # Get top-k
        k = min(self.k, len(similarities))
        top_indices = np.argpartition(similarities, -k)[-k:]
        top_indices = top_indices[np.argsort(-similarities[top_indices])]

        # Convert similarity to distance (1 - cosine_sim)
        distances = 1 - similarities[top_indices]

        return top_indices, distances

    def _distance_weights(self, distances: np.ndarray) -> np.ndarray:
        """Convert distances to weights (closer = higher weight)."""
        # Inverse distance weighting with epsilon to avoid division by zero
        eps = 1e-6
        weights = 1.0 / (distances + eps)
        weights /= weights.sum()
        return weights

    def save(self, output_dir: str):
        """Save model to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "embedding_model_name": self.embedding_model_name,
            "k": self.k,
            "embeddings": self.embeddings,
            "model_names": self.model_names,
            "output_lengths": self.output_lengths,
            "quality_scores": self.quality_scores,
            "similarity_scores": self.similarity_scores,
            "llm_judge_scores": self.llm_judge_scores,
        }
        with open(output_path / "knn_estimator.pkl", "wb") as f:
            pickle.dump(state, f)

        # Also save metadata as JSON for inspection
        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "k": self.k,
            "num_examples": len(self.embeddings) if self.embeddings is not None else 0,
            "embedding_dim": self.embeddings.shape[1] if self.embeddings is not None else 0,
            "model_names": self.model_names,
        }
        with open(output_path / "knn_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"KNN estimator saved to {output_path}")

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu") -> "KNNEstimator":
        """Load model from disk."""
        model_path = Path(model_dir) / "knn_estimator.pkl"
        with open(model_path, "rb") as f:
            state = pickle.load(f)

        estimator = cls(
            embedding_model_name=state["embedding_model_name"],
            k=state["k"],
            device=device,
        )
        estimator.embeddings = state["embeddings"]
        estimator.model_names = state["model_names"]
        estimator.output_lengths = state["output_lengths"]
        estimator.quality_scores = state["quality_scores"]
        estimator.similarity_scores = state.get("similarity_scores", {})
        estimator.llm_judge_scores = state.get("llm_judge_scores", {})

        logger.info(
            f"KNN estimator loaded: {len(estimator.embeddings)} examples, "
            f"{len(estimator.model_names)} models"
        )
        return estimator
