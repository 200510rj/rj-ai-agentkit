"""
Unit tests for agentkit types and Pydantic models.
"""

import pytest
from pydantic import ValidationError
from agentkit.types import (
    BinaryResult,
    ChoiceResult,
    ScoreResult,
    Violation,
    GuardResult,
    RouteConfig,
    RouteResult,
    GateResult,
    JudgeResult,
    SearchResult,
)


def test_binary_result_valid():
    res = BinaryResult(result=True, confidence=0.95, reasoning="Injection detected")
    assert res.result is True
    assert res.confidence == 0.95
    assert res.reasoning == "Injection detected"


def test_binary_result_invalid_confidence():
    with pytest.raises(ValidationError):
        BinaryResult(result=True, confidence=1.5)

    with pytest.raises(ValidationError):
        BinaryResult(result=True, confidence=-0.1)


def test_choice_result_valid():
    res = ChoiceResult(
        selected="billing",
        confidence=0.88,
        scores={"billing": 0.88, "tech": 0.12},
        reasoning="Matched billing query",
    )
    assert res.selected == "billing"
    assert res.scores["billing"] == 0.88


def test_choice_result_invalid_scores():
    with pytest.raises(ValidationError):
        ChoiceResult(selected="billing", confidence=0.9, scores={"billing": 1.2})


def test_score_result_valid():
    res = ScoreResult(
        score=0.85,
        criteria_scores={"relevance": 0.9, "clarity": 0.8},
        confidence=0.95,
        reasoning="Good answer",
    )
    assert res.score == 0.85
    assert res.criteria_scores["relevance"] == 0.9


def test_guard_result_immutability():
    violation = Violation(check_name="pii", severity="high", confidence=0.9)
    res = GuardResult(verdict="block", confidence=0.9, violations=[violation])
    with pytest.raises(ValidationError):
        res.verdict = "allow"  # type: ignore


def test_route_config_creation():
    cfg = RouteConfig(name="billing", description="Billing issues", handler=lambda q: "handled")
    assert cfg.name == "billing"
    assert cfg.handler("test") == "handled"


def test_gate_result_valid():
    res = GateResult(should_retrieve=True, verdict="retrieve", confidence=0.95, reasoning="Need facts")
    assert res.should_retrieve is True
    assert res.verdict == "retrieve"


def test_judge_result_valid():
    res = JudgeResult(
        verdict="pass",
        score=0.82,
        confidence=0.9,
        criteria_scores={"accuracy": 0.85},
        feedback="Keep up the clear explanations.",
    )
    assert res.verdict == "pass"
    assert res.feedback != ""
