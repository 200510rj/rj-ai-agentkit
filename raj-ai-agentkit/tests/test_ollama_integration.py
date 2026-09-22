"""
Unit tests for Ollama local provider integration (mocked HTTP / provider calls).
Does NOT require an active local Ollama server.
"""

from unittest.mock import MagicMock
import pytest
from agentkit import OpenAIProvider, LLMEngine, Guard, Router, RAGGate, Judge, RouteConfig
from agentkit.exceptions import ProviderError


def test_ollama_provider_configuration():
    mock_client = MagicMock()
    provider = OpenAIProvider(
        model="qwen3.5:4b",
        base_url="http://localhost:11434/v1",
        client=mock_client,
    )
    assert provider.model == "qwen3.5:4b"
    assert provider._is_custom_base_url is True


def test_ollama_structured_output_bypasses_openai_beta():
    """Verify custom base_url bypasses OpenAI cloud beta.parse and uses complete() + JSON parsing."""
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = '{"result": false, "confidence": 0.9, "reasoning": "Clean text"}'
    mock_client.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

    provider = OpenAIProvider(
        model="qwen3.5:4b",
        base_url="http://localhost:11434/v1",
        client=mock_client,
    )
    engine = LLMEngine(provider=provider)
    guard = Guard(engine=engine)

    res = guard.check("What is Python?")
    assert res.verdict == "allow"
    # Verify beta.chat.completions.parse was NOT called
    assert not mock_client.beta.chat.completions.parse.called
    # Verify chat.completions.create WAS called
    assert mock_client.chat.completions.create.called


def test_ollama_provider_error_handling():
    mock_client = MagicMock()
    mock_client.chat.completions.create.side_error = Exception("Connection refused")
    mock_client.chat.completions.create.side_effect = Exception("Connection refused")

    provider = OpenAIProvider(
        model="qwen3.5:4b",
        base_url="http://localhost:11434/v1",
        client=mock_client,
    )
    with pytest.raises(ProviderError):
        provider.complete("Test query")
