"""
Base interface for model quality scorers.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Tuple


class ModelScorer(ABC):
    """Base interface for model quality scoring.

    Scorers evaluate the quality of model responses given a prompt.
    """

    @abstractmethod
    def score(self,
              prompt: str,
              responses: List[Tuple[str, str]]) -> Dict[str, float]:
        """Compute quality scores for model responses.

        Args:
            prompt: Input prompt text
            responses: List of (model_name, generated_text) tuples

        Returns:
            Dict mapping model_name -> quality_score (0.0-1.0)
            where 1.0 indicates highest quality
        """
        pass