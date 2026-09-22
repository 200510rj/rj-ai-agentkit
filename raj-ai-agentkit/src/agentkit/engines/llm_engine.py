"""
LLMEngine: DecisionEngine powered by any LLMProvider.
"""

import json
import logging
from typing import Type, TypeVar
from pydantic import BaseModel
from agentkit.engine import DecisionEngine
from agentkit.provider import LLMProvider
from agentkit.types import BinaryResult, ChoiceResult, ScoreResult
from agentkit.exceptions import EngineError

logger = logging.getLogger("agentkit")

T = TypeVar("T", bound=BaseModel)


class LLMEngine(DecisionEngine):
    """
    LLM-based decision engine.
    Uses structured outputs via LLMProvider, with automatic JSON fallback parsing for local/smaller models.
    """

    def __init__(self, provider: LLMProvider, temperature: float = 0.0, max_retries: int = 2):
        self.provider = provider
        self.temperature = temperature
        self.max_retries = max_retries

    def _call_structured_with_fallback(self, prompt: str, schema: Type[T]) -> T:
        """Attempt provider.structured_output, falling back to complete() + json parsing on failure."""
        try:
            return self.provider.structured_output(prompt, schema, temperature=self.temperature)
        except Exception as err:
            logger.warning(f"Structured output call failed: {err}. Falling back to prompt + JSON parsing.")

        # Fallback path for local models (e.g. Ollama/vLLM) or raw providers
        schema_json = schema.model_json_schema()
        json_prompt = (
            f"{prompt}\n\n"
            f"IMPORTANT: Respond STRICTLY with a single valid JSON object matching this JSON Schema:\n"
            f"{json.dumps(schema_json, indent=2)}\n"
            f"Do not include markdown code block backticks (` ```json `) or any other conversational text."
        )

        last_error = None
        for attempt in range(self.max_retries + 1):
            try:
                raw_response = self.provider.complete(json_prompt, temperature=self.temperature)
                cleaned = raw_response.strip()
                start_idx = cleaned.find("{")
                end_idx = cleaned.rfind("}")
                if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                    cleaned = cleaned[start_idx : end_idx + 1]

                parsed_json = json.loads(cleaned)
                return schema.model_validate(parsed_json)
            except Exception as e:
                last_error = e
                logger.debug(f"JSON fallback parse attempt {attempt + 1} failed: {e}")

        raise EngineError(f"Failed to produce structured output conforming to {schema.__name__}: {last_error}")

    def check(self, text: str, instructions: str = "") -> BinaryResult:
        prompt = (
            f"You are a safety and verification gate for an AI agent.\n"
            f"Instruction: {instructions}\n\n"
            f"Input Text:\n\"\"\"{text}\"\"\"\n\n"
            f"Determine whether the input text satisfies or triggers the condition in the instruction."
        )
        return self._call_structured_with_fallback(prompt, BinaryResult)

    def classify(self, text: str, options: dict[str, str], instructions: str = "") -> ChoiceResult:
        options_formatted = "\n".join([f"- '{k}': {v}" for k, v in options.items()])
        prompt = (
            f"You are an intent router for an AI agent.\n"
            f"{f'Context / Instructions: {instructions}' if instructions else ''}\n\n"
            f"Available Options:\n{options_formatted}\n\n"
            f"Input Text:\n\"\"\"{text}\"\"\"\n\n"
            f"Select the single best matching option key from the options list above."
        )
        result = self._call_structured_with_fallback(prompt, ChoiceResult)
        if result.selected not in options:
            logger.warning(f"Engine returned unknown option '{result.selected}'. Falling back to matching option.")
            # Fallback to nearest matching option key or first key
            for key in options:
                if key.lower() in result.selected.lower():
                    return ChoiceResult(
                        selected=key,
                        confidence=result.confidence,
                        scores=result.scores,
                        reasoning=result.reasoning,
                    )
            first_key = list(options.keys())[0]
            return ChoiceResult(
                selected=first_key,
                confidence=0.1,
                scores=result.scores,
                reasoning=f"Unknown key '{result.selected}' returned by LLM; defaulted to '{first_key}'",
            )
        return result

    def score(self, text: str, criteria: list[str], instructions: str = "") -> ScoreResult:
        criteria_formatted = "\n".join([f"- {c}" for c in criteria])
        prompt = (
            f"You are a quality judge for an AI agent response.\n"
            f"{f'Context / Instructions: {instructions}' if instructions else ''}\n\n"
            f"Criteria to evaluate (score each from 0.0 to 1.0):\n{criteria_formatted}\n\n"
            f"Input Text to Evaluate:\n\"\"\"{text}\"\"\"\n\n"
            f"Provide an overall quality score (0.0 to 1.0), individual scores for each criterion, and reasoning."
        )
        return self._call_structured_with_fallback(prompt, ScoreResult)
