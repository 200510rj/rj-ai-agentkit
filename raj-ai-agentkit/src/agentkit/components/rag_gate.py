"""
RAGGate component: Gating decision layer for Retrieval-Augmented Generation.
"""

import json
import logging
from typing import Any
from agentkit.engine import DecisionEngine
from agentkit.types import GateResult

logger = logging.getLogger("agentkit")

DEFAULT_RAG_INSTRUCTIONS = (
    "Does this query require external knowledge retrieval to answer accurately? "
    "Answer YES if the query asks about specific facts, documents, data, or information "
    "that would not be in a general language model's training data. "
    "Answer NO if the query is conversational, asks for opinions, or can be answered from general knowledge."
)


class RAGGate:
    """
    Decides whether a query requires retrieval before calling LLM.
    Saves latency and vector database lookup costs.
    """

    def __init__(
        self,
        engine: DecisionEngine,
        instructions: str | None = None,
        threshold: float = 0.5,
    ):
        """
        Initialize RAGGate.

        Args:
            engine: The DecisionEngine backend.
            instructions: Custom gating prompt instructions (defaults to DEFAULT_RAG_INSTRUCTIONS).
            threshold: Confidence threshold for retrieval (0.0 to 1.0).
        """
        self.engine = engine
        self.instructions = instructions or DEFAULT_RAG_INSTRUCTIONS
        self.threshold = threshold

    def should_retrieve(self, query: str, context: dict[str, Any] | None = None) -> GateResult:
        """
        Determine whether retrieval is necessary for the given query.

        Args:
            query: User input query string.
            context: Optional context dictionary.

        Returns:
            GateResult with should_retrieve (bool), verdict ("retrieve" or "skip"), confidence, and reasoning.
        """
        if query is None or not isinstance(query, str):
            raise ValueError("query must be a string")

        if not query.strip():
            return GateResult(
                should_retrieve=False,
                verdict="skip",
                confidence=1.0,
                reasoning="Empty query; retrieval skipped by default",
            )

        full_instructions = self.instructions
        if context:
            full_instructions += f"\nContext: {json.dumps(context)}"

        logger.debug(f"RAGGate.should_retrieve called for query_len={len(query)}")

        result = self.engine.check(text=query, instructions=full_instructions)

        should_retrieve_bool = bool(result.result) and (result.confidence >= self.threshold)
        verdict_str = "retrieve" if should_retrieve_bool else "skip"

        logger.info(f"RAGGate verdict: {verdict_str}, confidence={result.confidence:.2f}")

        return GateResult(
            should_retrieve=should_retrieve_bool,
            verdict=verdict_str,
            confidence=result.confidence,
            reasoning=result.reasoning,
        )

    def __call__(self, query: str, **kwargs) -> GateResult:
        """Alias for should_retrieve()."""
        return self.should_retrieve(query, **kwargs)
