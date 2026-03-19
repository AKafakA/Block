#!/usr/bin/env python3
"""
MLP-based estimator for CARA output length and quality prediction.

Uses frozen sentence-transformer embeddings + per-model MLP heads.
Supports optional quantile regression for conservative length bounds (Q90/Q95).
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class PerModelMLP(nn.Module):
    """MLP head for a single model's length or quality prediction."""

    def __init__(self, input_dim: int, hidden_dims: List[int] = None, num_quantiles: int = 0):
        """
        Args:
            input_dim: Embedding dimension.
            hidden_dims: Hidden layer sizes. Default [256, 128].
            num_quantiles: If > 0, output extra quantile heads (e.g., 2 for Q90+Q95).
        """
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev_dim, h), nn.ReLU(), nn.Dropout(0.1)])
            prev_dim = h

        self.backbone = nn.Sequential(*layers)
        # Mean prediction head
        self.mean_head = nn.Linear(prev_dim, 1)
        # Optional quantile heads
        self.num_quantiles = num_quantiles
        if num_quantiles > 0:
            self.quantile_heads = nn.ModuleList(
                [nn.Linear(prev_dim, 1) for _ in range(num_quantiles)]
            )

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        h = self.backbone(x)
        out = {"mean": self.mean_head(h).squeeze(-1)}
        if self.num_quantiles > 0:
            out["quantiles"] = torch.stack(
                [qh(h).squeeze(-1) for qh in self.quantile_heads], dim=-1
            )
        return out


class MLPEstimator:
    """MLP-based length and quality estimator.

    Stores frozen sentence-transformer embeddings and trains per-model MLP heads
    for length prediction (with optional quantile regression) and quality prediction.
    """

    def __init__(
        self,
        embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        hidden_dims: List[int] = None,
        quantiles: List[float] = None,
        device: str = "cpu",
    ):
        self.embedding_model_name = embedding_model_name
        self.hidden_dims = hidden_dims or [256, 128]
        self.quantiles = quantiles or [0.9, 0.95]
        self.device = device

        self.model_names: List[str] = []
        self.length_models: Dict[str, PerModelMLP] = {}
        self.quality_models: Dict[str, PerModelMLP] = {}
        self.embedding_dim: int = 0

        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            logger.info(f"Loading embedding model: {self.embedding_model_name}")
            self._encoder = SentenceTransformer(
                self.embedding_model_name, device=self.device
            )
            self.embedding_dim = self._encoder.get_sentence_embedding_dimension()
            logger.info(f"Embedding model loaded: dim={self.embedding_dim}")
        return self._encoder

    @staticmethod
    def _extract_quality(m_data: Dict) -> float:
        """Extract combined quality score (same logic as KNNEstimator)."""
        if "quality_score" in m_data:
            return float(m_data["quality_score"])
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

    def train(
        self,
        training_data: List[Dict],
        epochs: int = 50,
        lr: float = 1e-3,
        batch_size: int = 256,
        val_split: float = 0.1,
    ) -> Dict[str, List[float]]:
        """Train MLP heads on preprocessed training data.

        Args:
            training_data: List of dicts with {prompt, models: {model: {output_length, ...}}}
            epochs: Training epochs.
            lr: Learning rate.
            batch_size: Training batch size.
            val_split: Fraction of data to use for validation.

        Returns:
            Training history with loss curves per model.
        """
        n = len(training_data)
        logger.info(f"Training MLP estimator on {n} examples...")

        # Encode all prompts
        encoder = self._get_encoder()
        prompts = [d["prompt"] for d in training_data]
        embeddings = encoder.encode(
            prompts, batch_size=64, show_progress_bar=True, normalize_embeddings=True
        )
        embeddings = torch.tensor(embeddings, dtype=torch.float32)
        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Collect model names
        self.model_names = sorted(training_data[0]["models"].keys())

        # Train/val split
        n_val = int(n * val_split)
        n_train = n - n_val
        indices = np.random.permutation(n)
        train_idx = indices[:n_train]
        val_idx = indices[n_train:]

        train_emb = embeddings[train_idx].to(self.device)
        val_emb = embeddings[val_idx].to(self.device)

        history = {}

        for model_name in self.model_names:
            logger.info(f"Training heads for {model_name}...")

            # Extract labels
            lengths = np.array([
                d["models"].get(model_name, {}).get("output_length", 0)
                for d in training_data
            ], dtype=np.float32)
            qualities = np.array([
                self._extract_quality(d["models"].get(model_name, {}))
                for d in training_data
            ], dtype=np.float32)

            train_lengths = torch.tensor(lengths[train_idx]).to(self.device)
            train_qualities = torch.tensor(qualities[train_idx]).to(self.device)
            val_lengths = torch.tensor(lengths[val_idx]).to(self.device)
            val_qualities = torch.tensor(qualities[val_idx]).to(self.device)

            # Create models
            length_mlp = PerModelMLP(
                self.embedding_dim, self.hidden_dims,
                num_quantiles=len(self.quantiles)
            ).to(self.device)
            quality_mlp = PerModelMLP(
                self.embedding_dim, self.hidden_dims, num_quantiles=0
            ).to(self.device)

            # Train length head
            length_losses = self._train_head(
                length_mlp, train_emb, train_lengths, val_emb, val_lengths,
                epochs=epochs, lr=lr, batch_size=batch_size,
                quantile_targets=self.quantiles,
            )
            # Train quality head
            quality_losses = self._train_head(
                quality_mlp, train_emb, train_qualities, val_emb, val_qualities,
                epochs=epochs, lr=lr, batch_size=batch_size,
            )

            self.length_models[model_name] = length_mlp
            self.quality_models[model_name] = quality_mlp

            history[model_name] = {
                "length_train_loss": length_losses["train"],
                "length_val_loss": length_losses["val"],
                "quality_train_loss": quality_losses["train"],
                "quality_val_loss": quality_losses["val"],
            }

            logger.info(
                f"  {model_name}: length_val_loss={length_losses['val'][-1]:.4f}, "
                f"quality_val_loss={quality_losses['val'][-1]:.6f}"
            )

        return history

    def _train_head(
        self,
        model: PerModelMLP,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        val_x: torch.Tensor,
        val_y: torch.Tensor,
        epochs: int,
        lr: float,
        batch_size: int,
        quantile_targets: List[float] = None,
    ) -> Dict[str, List[float]]:
        """Train a single MLP head."""
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        mse_loss = nn.MSELoss()

        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_state = None

        n_train = len(train_x)

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0.0
            n_batches = 0

            perm = torch.randperm(n_train, device=train_x.device)
            for i in range(0, n_train, batch_size):
                idx = perm[i : i + batch_size]
                x_batch = train_x[idx]
                y_batch = train_y[idx]

                out = model(x_batch)
                loss = mse_loss(out["mean"], y_batch)

                # Quantile loss
                if quantile_targets and "quantiles" in out:
                    for qi, q in enumerate(quantile_targets):
                        residual = y_batch - out["quantiles"][:, qi]
                        ql = torch.where(
                            residual >= 0, q * residual, (q - 1) * residual
                        )
                        loss = loss + ql.mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                n_batches += 1

            scheduler.step()
            train_losses.append(epoch_loss / n_batches)

            # Validation
            model.eval()
            with torch.no_grad():
                out = model(val_x)
                val_loss = mse_loss(out["mean"], val_y).item()
                val_losses.append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {k: v.clone() for k, v in model.state_dict().items()}

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        return {"train": train_losses, "val": val_losses}

    def predict_length(self, prompt: str, model_name: str) -> Dict[str, float]:
        """Predict output length for a prompt on a given model."""
        emb = self._embed_prompt(prompt)
        mlp = self.length_models[model_name]
        mlp.eval()
        with torch.no_grad():
            out = mlp(emb)
        result = {"mean": float(out["mean"].item())}
        if "quantiles" in out:
            for qi, q in enumerate(self.quantiles):
                key = f"p{int(q * 100)}"
                result[key] = float(out["quantiles"][0, qi].item())
        return result

    def predict_quality(self, prompt: str, model_name: str) -> float:
        """Predict quality score for a prompt on a given model."""
        emb = self._embed_prompt(prompt)
        mlp = self.quality_models[model_name]
        mlp.eval()
        with torch.no_grad():
            out = mlp(emb)
        return float(out["mean"].item())

    def predict_all_models(self, prompt: str) -> Dict[str, Dict[str, float]]:
        """Predict length and quality for all models at once."""
        emb = self._embed_prompt(prompt)
        results = {}
        for model_name in self.model_names:
            self.length_models[model_name].eval()
            self.quality_models[model_name].eval()
            with torch.no_grad():
                length_out = self.length_models[model_name](emb)
                quality_out = self.quality_models[model_name](emb)

            result = {
                "length_mean": float(length_out["mean"].item()),
                "quality_score": float(quality_out["mean"].item()),
            }
            if "quantiles" in length_out:
                for qi, q in enumerate(self.quantiles):
                    result[f"length_p{int(q * 100)}"] = float(
                        length_out["quantiles"][0, qi].item()
                    )
            results[model_name] = result
        return results

    def _embed_prompt(self, prompt: str) -> torch.Tensor:
        """Embed a single prompt and return as tensor."""
        encoder = self._get_encoder()
        emb = encoder.encode([prompt], normalize_embeddings=True)
        return torch.tensor(emb, dtype=torch.float32).to(self.device)

    def save(self, output_dir: str):
        """Save model to disk."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        state = {
            "embedding_model_name": self.embedding_model_name,
            "hidden_dims": self.hidden_dims,
            "quantiles": self.quantiles,
            "model_names": self.model_names,
            "embedding_dim": self.embedding_dim,
            "length_models": {
                m: mlp.state_dict() for m, mlp in self.length_models.items()
            },
            "quality_models": {
                m: mlp.state_dict() for m, mlp in self.quality_models.items()
            },
        }
        torch.save(state, output_path / "mlp_estimator.pt")

        metadata = {
            "embedding_model_name": self.embedding_model_name,
            "hidden_dims": self.hidden_dims,
            "quantiles": self.quantiles,
            "model_names": self.model_names,
            "embedding_dim": self.embedding_dim,
        }
        with open(output_path / "mlp_metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"MLP estimator saved to {output_path}")

    @classmethod
    def load(cls, model_dir: str, device: str = "cpu") -> "MLPEstimator":
        """Load model from disk."""
        model_path = Path(model_dir) / "mlp_estimator.pt"
        state = torch.load(model_path, map_location=device, weights_only=False)

        estimator = cls(
            embedding_model_name=state["embedding_model_name"],
            hidden_dims=state["hidden_dims"],
            quantiles=state["quantiles"],
            device=device,
        )
        estimator.model_names = state["model_names"]
        estimator.embedding_dim = state["embedding_dim"]

        for model_name in estimator.model_names:
            length_mlp = PerModelMLP(
                estimator.embedding_dim, estimator.hidden_dims,
                num_quantiles=len(estimator.quantiles)
            ).to(device)
            length_mlp.load_state_dict(state["length_models"][model_name])
            estimator.length_models[model_name] = length_mlp

            quality_mlp = PerModelMLP(
                estimator.embedding_dim, estimator.hidden_dims, num_quantiles=0
            ).to(device)
            quality_mlp.load_state_dict(state["quality_models"][model_name])
            estimator.quality_models[model_name] = quality_mlp

        logger.info(
            f"MLP estimator loaded: {len(estimator.model_names)} models, "
            f"dim={estimator.embedding_dim}"
        )
        return estimator
