"""
OpenAIProvider: Implementation of LLMProvider for OpenAI and OpenAI-compatible local APIs (Ollama, vLLM, LMStudio).
Lazy-imports `openai` so core agentkit never requires the openai library.
"""

import os
import json
import logging
from typing import Type, TypeVar, Any
from pydantic import BaseModel
from agentkit.provider import LLMProvider
from agentkit.exceptions import ProviderError

logger = logging.getLogger("agentkit")

T = TypeVar("T", bound=BaseModel)


class OpenAIProvider(LLMProvider):
    """
    LLMProvider for OpenAI and OpenAI-compatible APIs (Ollama, vLLM, LM Studio, LocalAI).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        client: Any | None = None,
    ):
        """
        Initialize OpenAIProvider.

        Args:
            model: Model ID (e.g. "gpt-4o-mini", "gpt-4o", "llama3.2:3b").
            api_key: OpenAI API key (reads OPENAI_API_KEY env var if None).
            base_url: Override base URL for Ollama/vLLM (reads OPENAI_BASE_URL env var if None).
            client: Pre-instantiated openai.OpenAI client (optional, for testing/mocking).
        """
        self.model = model
        resolved_api_key = api_key or os.environ.get("OPENAI_API_KEY") or "dummy-key-for-local"
        resolved_base_url = base_url or os.environ.get("OPENAI_BASE_URL")
        self._is_custom_base_url = bool(resolved_base_url)

        if client is not None:
            self._client = client
        else:
            try:
                from openai import OpenAI  # type: ignore
            except ImportError:
                raise ImportError(
                    "OpenAI provider requires the openai package. Install it with: pip install raj-ai-agentkit[openai]"
                )

            client_kwargs: dict[str, Any] = {"api_key": resolved_api_key}
            if resolved_base_url:
                client_kwargs["base_url"] = resolved_base_url

            try:
                self._client = OpenAI(**client_kwargs)
            except Exception as e:
                raise ProviderError(f"Failed to initialize OpenAI client: {e}") from e

    def complete(self, prompt: str, **kwargs) -> str:
        """Generate raw text completion."""
        try:
            temperature = kwargs.get("temperature", 0.0)
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            content = response.choices[0].message.content
            return content or ""
        except Exception as e:
            raise ProviderError(f"OpenAI complete call failed: {e}") from e

    def structured_output(self, prompt: str, schema: Type[T], **kwargs) -> T:
        """
        Generate response conforming strictly to a Pydantic schema using response_format parsing,
        with fallback to complete() + Pydantic validation for local/non-OpenAI models.
        """
        temperature = kwargs.get("temperature", 0.0)

        # Attempt native beta.chat.completions.parse structured output (OpenAI cloud only)
        if not getattr(self, "_is_custom_base_url", False):
            try:
                response = self._client.beta.chat.completions.parse(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    response_format=schema,
                    temperature=temperature,
                )
                parsed_message = response.choices[0].message
                if getattr(parsed_message, "parsed", None) is not None:
                    parsed_obj = parsed_message.parsed
                    if isinstance(parsed_obj, schema):
                        return parsed_obj
            except Exception as e:
                logger.debug(f"Native structured_output failed: {e}. Attempting JSON mode / completion fallback.")

        # Fallback path: completion with JSON schema instructions
        try:
            schema_json = schema.model_json_schema()
            json_prompt = (
                f"{prompt}\n\n"
                f"Respond STRICTLY with valid JSON matching this schema:\n"
                f"{json.dumps(schema_json, indent=2)}\n"
                f"Return JSON ONLY."
            )
            raw_text = self.complete(json_prompt, temperature=temperature)
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[-1]
            if cleaned.endswith("```"):
                cleaned = cleaned.rsplit("\n", 1)[0]
            if cleaned.startswith("json"):
                cleaned = cleaned[4:].strip()

            return schema.model_validate_json(cleaned)
        except Exception as err:
            raise ProviderError(f"OpenAI structured output generation and parsing failed: {err}") from err
