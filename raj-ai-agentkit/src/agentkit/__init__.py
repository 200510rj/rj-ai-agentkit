"""
raj-ai-agentkit: A reusable, provider-agnostic Python toolkit for AI agent reliability primitives.
"""

from agentkit.components import Guard, Router, RAGGate, Judge
from agentkit.engines import RuleEngine, LLMEngine, LayaEngine, Rule, KeywordRule, RegexRule, CallableRule
from agentkit.providers import OpenAIProvider
from agentkit.engine import DecisionEngine
from agentkit.provider import LLMProvider, VectorStore
from agentkit.types import (
    GuardResult,
    RouteResult,
    GateResult,
    JudgeResult,
    RouteConfig,
    SearchResult,
    Violation,
    BinaryResult,
    ChoiceResult,
    ScoreResult,
)
from agentkit.exceptions import (
    AgentKitError,
    EngineError,
    ProviderError,
    ConfigError,
)

__version__ = "0.1.0"

__all__ = [
    # Core Components
    "Guard",
    "Router",
    "RAGGate",
    "Judge",
    # Engines
    "RuleEngine",
    "LLMEngine",
    "LayaEngine",
    "Rule",
    "KeywordRule",
    "RegexRule",
    "CallableRule",
    # Providers
    "OpenAIProvider",
    # Base Abstractions
    "DecisionEngine",
    "LLMProvider",
    "VectorStore",
    # Data Models
    "GuardResult",
    "RouteResult",
    "GateResult",
    "JudgeResult",
    "RouteConfig",
    "SearchResult",
    "Violation",
    "BinaryResult",
    "ChoiceResult",
    "ScoreResult",
    # Exceptions
    "AgentKitError",
    "EngineError",
    "ProviderError",
    "ConfigError",
]
