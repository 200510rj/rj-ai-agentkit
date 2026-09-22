"""
Custom exceptions for agentkit.
"""

class AgentKitError(Exception):
    """Base exception for all agentkit errors."""
    pass


class EngineError(AgentKitError):
    """Raised when a decision engine fails to produce a valid result."""
    pass


class ProviderError(AgentKitError):
    """Raised when an LLM or vector store provider fails."""
    pass


class ConfigError(AgentKitError):
    """Raised for invalid component or engine configuration."""
    pass
