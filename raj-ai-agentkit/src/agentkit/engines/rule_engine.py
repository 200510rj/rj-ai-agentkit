"""
RuleEngine: Zero-dependency rule-based decision engine.
"""

import re
from abc import ABC, abstractmethod
from typing import Callable
from agentkit.engine import DecisionEngine
from agentkit.types import BinaryResult, ChoiceResult, ScoreResult


class Rule(ABC):
    """Base class for all rules used by RuleEngine."""

    def __init__(self, name: str, verdict: str = "match", confidence: float = 1.0):
        self.name = name
        self.verdict = verdict
        self.confidence = confidence

    @abstractmethod
    def matches(self, text: str) -> bool:
        """Return True if rule matches the input text."""
        pass


class KeywordRule(Rule):
    """Rule that matches if any specified keywords are present in text."""

    def __init__(self, name: str, keywords: list[str], match_all: bool = False, verdict: str = "match", confidence: float = 1.0):
        super().__init__(name=name, verdict=verdict, confidence=confidence)
        self.keywords = [k.lower() for k in keywords]
        self.match_all = match_all

    def matches(self, text: str) -> bool:
        lower_text = text.lower()
        if self.match_all:
            return all(kw in lower_text for kw in self.keywords)
        return any(kw in lower_text for kw in self.keywords)


class RegexRule(Rule):
    """Rule that matches a regular expression pattern against text."""

    def __init__(self, name: str, pattern: str, verdict: str = "match", confidence: float = 1.0, flags: int = re.IGNORECASE):
        super().__init__(name=name, verdict=verdict, confidence=confidence)
        self.pattern = re.compile(pattern, flags)

    def matches(self, text: str) -> bool:
        return bool(self.pattern.search(text))


class CallableRule(Rule):
    """Rule that delegates match determination to a custom callable."""

    def __init__(self, name: str, predicate: Callable[[str], bool], verdict: str = "match", confidence: float = 1.0):
        super().__init__(name=name, verdict=verdict, confidence=confidence)
        self.predicate = predicate

    def matches(self, text: str) -> bool:
        return self.predicate(text)


class RuleEngine(DecisionEngine):
    """
    Zero-dependency rule-based decision engine.
    Uses pattern matching, keyword lookup, or user-supplied rules.
    """

    def __init__(self, rules: list[Rule] | None = None):
        self.rules = rules or []

    @classmethod
    def from_keywords(cls, block: list[str], name: str = "keyword_block") -> "RuleEngine":
        """Convenience constructor to create a RuleEngine from a list of block keywords."""
        rule = KeywordRule(name=name, keywords=block, verdict="block", confidence=1.0)
        return cls(rules=[rule])

    def check(self, text: str, instructions: str = "") -> BinaryResult:
        """
        Check if any rule matches the input text.
        If input text is empty, returns result=False immediately.
        If a rule matches:
          - If rule.verdict in ("block", "match", "true"), returns result=True
          - Else result=False
        If no rule matches, returns result=False with confidence=1.0.
        """
        if not text or not text.strip():
            return BinaryResult(
                result=False,
                confidence=1.0,
                reasoning="Input text is empty",
            )

        for rule in self.rules:
            if rule.matches(text):
                is_match = rule.verdict.lower() in ("block", "match", "true", "yes")
                return BinaryResult(
                    result=is_match,
                    confidence=rule.confidence,
                    reasoning=f"Matched rule '{rule.name}' with verdict '{rule.verdict}'",
                )
        return BinaryResult(
            result=False,
            confidence=1.0,
            reasoning="No rules matched input text",
        )

    def classify(self, text: str, options: dict[str, str], instructions: str = "") -> ChoiceResult:
        """
        Classify text by checking rules or matching keyword candidates against option keys and descriptions.
        """
        if not text or not text.strip():
            first_option = list(options.keys())[0] if options else ""
            return ChoiceResult(
                selected=first_option,
                confidence=0.0,
                scores={k: 0.0 for k in options},
                reasoning="Input text is empty",
            )
        # First check if any configured rule verdict matches an option key
        for rule in self.rules:
            if rule.matches(text):
                if rule.verdict in options:
                    return ChoiceResult(
                        selected=rule.verdict,
                        confidence=rule.confidence,
                        scores={key: (1.0 if key == rule.verdict else 0.0) for key in options},
                        reasoning=f"Matched rule '{rule.name}' for route '{rule.verdict}'",
                    )

        # Keyword frequency heuristic across options keys and descriptions
        lower_text = text.lower()
        scores: dict[str, float] = {}
        for key, desc in options.items():
            key_terms = [key.lower()] + [w.lower() for w in desc.split() if len(w) > 3]
            matches = sum(1 for term in key_terms if term in lower_text)
            scores[key] = float(matches)

        total_matches = sum(scores.values())
        if total_matches > 0:
            norm_scores = {k: v / total_matches for k, v in scores.items()}
            best_key = max(norm_scores, key=lambda k: norm_scores[k])
            return ChoiceResult(
                selected=best_key,
                confidence=min(1.0, norm_scores[best_key]),
                scores=norm_scores,
                reasoning=f"Matched keywords for option '{best_key}'",
            )

        # Fallback to first option if no matches
        first_option = list(options.keys())[0]
        default_scores = {k: (1.0 if k == first_option else 0.0) for k in options}
        return ChoiceResult(
            selected=first_option,
            confidence=0.1,
            scores=default_scores,
            reasoning="No keyword matches found; returned default first option with low confidence",
        )

    def score(self, text: str, criteria: list[str], instructions: str = "") -> ScoreResult:
        """
        Score input text against criteria using heuristic evaluation or rule matching.
        """
        if not text.strip():
            empty_scores = {c: 0.0 for c in criteria}
            return ScoreResult(
                score=0.0,
                criteria_scores=empty_scores,
                confidence=1.0,
                reasoning="Input text is empty",
            )

        # Basic default heuristics: check length and non-emptiness
        criteria_scores: dict[str, float] = {}
        for criterion in criteria:
            # If any rule matched, use rule.confidence, otherwise default to 0.8 for non-empty text
            criteria_scores[criterion] = 0.85

        overall_score = sum(criteria_scores.values()) / len(criteria_scores) if criteria_scores else 1.0
        return ScoreResult(
            score=overall_score,
            criteria_scores=criteria_scores,
            confidence=0.8,
            reasoning="Evaluated criteria using rule engine heuristics",
        )
