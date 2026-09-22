"""
LayaEngine: DecisionEngine powered by Laya (~33ms non-autoregressive decision model).
Lazy-imports `laya` so agentkit startup never requires PyTorch or Transformers unless LayaEngine is instantiated.
"""

import logging
from typing import Any
from agentkit.engine import DecisionEngine
from agentkit.types import BinaryResult, ChoiceResult, ScoreResult
from agentkit.exceptions import EngineError

logger = logging.getLogger("agentkit")


class LayaEngine(DecisionEngine):
    """
    DecisionEngine implementation backed by Laya.
    Provides ~33ms non-autoregressive decision latency.
    """

    def __init__(self, model: str = "default", device: str = "cpu", preload: bool = True, router_instance: Any = None):
        """
        Initialize LayaEngine.

        Args:
            model: Laya model checkpoint name or path.
            device: Computing device ('cpu', 'cuda', etc.).
            preload: Whether to load model checkpoint immediately on initialization.
            router_instance: Pre-instantiated laya.Router object (optional, for testing/injection).
        """
        if router_instance is not None:
            self._router = router_instance
        else:
            try:
                from laya import Router as LayaRouter  # type: ignore
            except ImportError:
                raise ImportError(
                    "Laya is an optional dependency. Install it with: pip install raj-ai-agentkit[laya]"
                )
            self._router = LayaRouter(model=model, device=device, preload=preload)

    def check(self, text: str, instructions: str = "") -> BinaryResult:
        """Translates binary check to Laya 'noul' (yes/no) question format."""
        try:
            question_spec = {
                "type": "noul",
                "instructions": instructions or "Does the text satisfy the check condition?",
            }
            response = self._router.predict(text=text, question=question_spec)
            
            # Extract answers from Laya response format
            answer = response.get("answer", {}) if isinstance(response, dict) else getattr(response, "answer", {})
            result_val = bool(answer.get("noul", False))
            confidence_val = float(response.get("confidence", 0.95))
            reasoning_val = str(response.get("reasoning", "Decided by Laya noul model"))

            return BinaryResult(
                result=result_val,
                confidence=confidence_val,
                reasoning=reasoning_val,
            )
        except Exception as e:
            raise EngineError(f"LayaEngine check failed: {e}") from e

    def classify(self, text: str, options: dict[str, str], instructions: str = "") -> ChoiceResult:
        """Translates intent classification to Laya 'choice' question format."""
        try:
            question_spec = {
                "type": "choice",
                "instructions": instructions or "Select the best route matching the input text.",
                "criteria": options,
            }
            response = self._router.predict(text=text, question=question_spec)

            answer = response.get("answer", {}) if isinstance(response, dict) else getattr(response, "answer", {})
            selected_key = str(answer.get("choice", list(options.keys())[0]))
            confidence_val = float(response.get("confidence", 0.95))
            scores_map = response.get("scores", {k: (1.0 if k == selected_key else 0.0) for k in options})
            reasoning_val = str(response.get("reasoning", "Decided by Laya choice model"))

            if selected_key not in options:
                selected_key = list(options.keys())[0]

            return ChoiceResult(
                selected=selected_key,
                confidence=confidence_val,
                scores=scores_map,
                reasoning=reasoning_val,
            )
        except Exception as e:
            raise EngineError(f"LayaEngine classify failed: {e}") from e

    def score(self, text: str, criteria: list[str], instructions: str = "") -> ScoreResult:
        """Translates quality scoring to Laya 'score' question format."""
        try:
            question_spec = {
                "type": "score",
                "instructions": instructions or "Score the input text quality across the given criteria.",
                "criteria": criteria,
            }
            response = self._router.predict(text=text, question=question_spec)

            answer = response.get("answer", {}) if isinstance(response, dict) else getattr(response, "answer", {})
            score_val = float(answer.get("score", 0.8))
            criteria_map = response.get("criteria_scores", {c: score_val for c in criteria})
            confidence_val = float(response.get("confidence", 0.95))
            reasoning_val = str(response.get("reasoning", "Scored by Laya score model"))

            return ScoreResult(
                score=score_val,
                criteria_scores=criteria_map,
                confidence=confidence_val,
                reasoning=reasoning_val,
            )
        except Exception as e:
            raise EngineError(f"LayaEngine score failed: {e}") from e
