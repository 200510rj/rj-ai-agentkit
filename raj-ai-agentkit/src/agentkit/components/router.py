"""
Router component: Intent classification and route selection layer.
"""

import json
import logging
from typing import Any
from agentkit.engine import DecisionEngine
from agentkit.types import RouteConfig, RouteResult
from agentkit.exceptions import ConfigError

logger = logging.getLogger("agentkit")


class Router:
    """
    Routes incoming user queries to the appropriate handler, LLM, or tool.
    Uses DecisionEngine.classify() to evaluate route options.
    """

    def __init__(
        self,
        engine: DecisionEngine,
        routes: dict[str, RouteConfig],
        fallback: str | None = None,
        confidence_threshold: float = 0.3,
    ):
        """
        Initialize Router.

        Args:
            engine: The DecisionEngine backend.
            routes: Dict mapping route_key -> RouteConfig. Must have at least 2 entries.
            fallback: Route key to use if confidence is below confidence_threshold.
            confidence_threshold: Minimum confidence required (default: 0.3).
        """
        if len(routes) < 2:
            raise ConfigError("Router requires at least 2 routes")

        if fallback is not None and fallback not in routes:
            raise ConfigError(f"Fallback route '{fallback}' is not in configured routes")

        self.engine = engine
        self.routes = routes
        self.fallback = fallback
        self.confidence_threshold = confidence_threshold

    def route(self, query: str, context: dict[str, Any] | None = None) -> RouteResult:
        """
        Route query to the best destination.

        Args:
            query: The input user query.
            context: Optional context dictionary.

        Returns:
            RouteResult with verdict (route key), confidence, scores, and reasoning.
        """
        if query is None or not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a non-empty string")

        options = {key: config.description for key, config in self.routes.items()}
        instructions = f"Context: {json.dumps(context)}" if context else ""

        logger.debug(f"Router.route called: query_len={len(query)}, routes={list(options.keys())}")

        choice = self.engine.classify(text=query, options=options, instructions=instructions)

        verdict_key = choice.selected
        confidence_val = choice.confidence

        if confidence_val < self.confidence_threshold and self.fallback:
            logger.info(f"Confidence {confidence_val:.2f} below threshold {self.confidence_threshold}; using fallback '{self.fallback}'")
            verdict_key = self.fallback

        logger.info(f"Route selected: {verdict_key}, confidence={confidence_val:.2f}")

        return RouteResult(
            verdict=verdict_key,
            confidence=confidence_val,
            scores=choice.scores,
            reasoning=choice.reasoning,
        )

    def route_and_execute(self, query: str, context: dict[str, Any] | None = None) -> tuple[RouteResult, Any]:
        """
        Routes the query and executes the handler attached to the selected RouteConfig (if available).

        Returns:
            Tuple of (RouteResult, handler_output_or_None)
        """
        result = self.route(query=query, context=context)
        route_config = self.routes.get(result.verdict)
        output = None
        if route_config and route_config.handler is not None:
            output = route_config.handler(query)
        return result, output

    def __call__(self, query: str, **kwargs) -> RouteResult:
        """Alias for route()."""
        return self.route(query, **kwargs)
