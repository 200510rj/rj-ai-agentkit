"""
DecisionEngine abstract base class.
"""

from abc import ABC, abstractmethod
from agentkit.types import BinaryResult, ChoiceResult, ScoreResult


class DecisionEngine(ABC):
    """
    Abstract base class for all decision backends.

    Every decision engine must implement three primitive methods:
      - classify(): pick one option from labeled candidates (used by Router)
      - score(): evaluate input on a 0.0-1.0 scale across criteria (used by Judge)
      - check(): binary yes/no decision (used by Guard and RAGGate)
    """

    @abstractmethod
    def classify(
        self,
        text: str,
        options: dict[str, str],
        instructions: str = "",
    ) -> ChoiceResult:
        """
        Pick one option from a labeled set of options.

        Args:
            text: Input text to classify.
            options: Mapping of option key -> option description.
            instructions: Optional context or guidance for classification.

        Returns:
            ChoiceResult containing the selected option key, confidence, and reasoning.
        """
        pass

    @abstractmethod
    def score(
        self,
        text: str,
        criteria: list[str],
        instructions: str = "",
    ) -> ScoreResult:
        """
        Score input text on a 0.0-1.0 scale against given criteria.

        Args:
            text: Input text to evaluate.
            criteria: List of criteria names.
            instructions: Optional context or guidance for scoring.

        Returns:
            ScoreResult containing overall score, criteria breakdown, and reasoning.
        """
        pass

    @abstractmethod
    def check(
        self,
        text: str,
        instructions: str = "",
    ) -> BinaryResult:
        """
        Perform a binary (yes/no) check on input text.

        Args:
            text: Input text to check.
            instructions: Plain-English instruction describing what to check.

        Returns:
            BinaryResult with result=True/False, confidence, and reasoning.
        """
        pass
