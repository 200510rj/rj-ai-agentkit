"""
Judge component: Response quality evaluation layer.
"""

import logging
from agentkit.engine import DecisionEngine
from agentkit.types import JudgeResult
from agentkit.exceptions import ConfigError

logger = logging.getLogger("agentkit")

DEFAULT_CRITERIA = ["relevance", "accuracy", "completeness", "clarity"]


class Judge:
    """
    Evaluates the quality of agent responses.
    Provides scoring, pass/fail/borderline verdict assignment, feedback generation for retries, and pairwise response comparisons.
    """

    def __init__(
        self,
        engine: DecisionEngine,
        criteria: list[str] | None = None,
        pass_threshold: float = 0.6,
        fail_threshold: float = 0.3,
    ):
        """
        Initialize Judge.

        Args:
            engine: The DecisionEngine backend.
            criteria: List of evaluation criteria (defaults to DEFAULT_CRITERIA).
            pass_threshold: Minimum score required for "pass" (default: 0.6).
            fail_threshold: Scores <= this yield "fail" (default: 0.3).
        """
        if pass_threshold <= fail_threshold:
            raise ConfigError("pass_threshold must be strictly greater than fail_threshold")

        self.engine = engine
        self.criteria = criteria if criteria is not None else list(DEFAULT_CRITERIA)
        self.pass_threshold = pass_threshold
        self.fail_threshold = fail_threshold

    def score(self, query: str, response: str, context: str = "") -> JudgeResult:
        """
        Evaluate and score a single agent response to a query.

        Args:
            query: The original user query.
            response: The agent's generated response.
            context: Optional background context or ground truth.

        Returns:
            JudgeResult containing score, verdict, criteria breakdown, and actionable feedback.
        """
        if response is None or not isinstance(response, str) or not response.strip():
            empty_criteria = {c: 0.0 for c in self.criteria}
            return JudgeResult(
                verdict="fail",
                score=0.0,
                confidence=1.0,
                criteria_scores=empty_criteria,
                feedback="The agent provided an empty or null response.",
                reasoning="Response is empty",
            )

        eval_text = f"Query: {query}\nResponse: {response}"
        if context:
            eval_text += f"\nContext: {context}"

        logger.debug(f"Judge.score called for query_len={len(query)}, response_len={len(response)}")

        result = self.engine.score(text=eval_text, criteria=self.criteria)

        score_val = result.score
        if score_val >= self.pass_threshold:
            verdict_str = "pass"
        elif score_val <= self.fail_threshold:
            verdict_str = "fail"
        else:
            verdict_str = "borderline"

        # Generate actionable feedback for low scoring criteria
        low_criteria = [
            f"Improve {c} (scored {val:.2f})"
            for c, val in result.criteria_scores.items()
            if val < self.pass_threshold
        ]
        feedback_str = "; ".join(low_criteria) if low_criteria else "Response met quality standard."

        logger.info(f"Judge score: {score_val:.2f}, verdict={verdict_str}")

        return JudgeResult(
            verdict=verdict_str,
            score=score_val,
            confidence=result.confidence,
            criteria_scores=result.criteria_scores,
            feedback=feedback_str,
            reasoning=result.reasoning,
        )

    def compare(
        self,
        query: str,
        response_a: str,
        response_b: str,
        context: str = "",
    ) -> tuple[JudgeResult, JudgeResult, str]:
        """
        Compare two candidate agent responses.

        Returns:
            Tuple of (JudgeResult_A, JudgeResult_B, winner_key) where winner_key is "a", "b", or "tie".
        """
        result_a = self.score(query=query, response=response_a, context=context)
        result_b = self.score(query=query, response=response_b, context=context)

        if abs(result_a.score - result_b.score) < 0.05:
            winner = "tie"
        elif result_a.score > result_b.score:
            winner = "a"
        else:
            winner = "b"

        return result_a, result_b, winner

    def __call__(self, query: str, response: str, **kwargs) -> JudgeResult:
        """Alias for score()."""
        return self.score(query, response, **kwargs)
