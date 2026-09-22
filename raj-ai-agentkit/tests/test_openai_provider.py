"""
Unit tests for OpenAIProvider (mocked client).
"""

from unittest.mock import MagicMock
from agentkit.providers.openai_provider import OpenAIProvider
from agentkit.types import BinaryResult


def test_openai_provider_mock_complete():
    mock_client = MagicMock()
    mock_choice = MagicMock()
    mock_choice.message.content = "Mocked LLM Response"
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.chat.completions.create.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)
    res = provider.complete("Test prompt")
    assert res == "Mocked LLM Response"


def test_openai_provider_mock_structured_output():
    mock_client = MagicMock()
    mock_parsed_obj = BinaryResult(result=True, confidence=0.99, reasoning="Mock parsed")
    mock_choice = MagicMock()
    mock_choice.message.parsed = mock_parsed_obj
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_client.beta.chat.completions.parse.return_value = mock_response

    provider = OpenAIProvider(client=mock_client)
    res = provider.structured_output("Test prompt", BinaryResult)
    assert res.result is True
    assert res.confidence == 0.99
