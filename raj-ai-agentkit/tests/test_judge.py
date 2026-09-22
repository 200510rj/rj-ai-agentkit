"""
Unit tests for Judge component.
"""

import pytest
from agentkit.components.judge import Judge
from agentkit.engines.rule_engine import RuleEngine
from agentkit.exceptions import ConfigError


def test_judge_score_pass():
    engine = RuleEngine()
    judge = Judge(engine=engine, pass_threshold=0.6, fail_threshold=0.3)
    res = judge.score(query="What is 2+2?", response="2 + 2 equals 4.")
    assert res.verdict == "pass"
    assert res.score >= 0.6


def test_judge_empty_response():
    engine = RuleEngine()
    judge = Judge(engine=engine)
    res = judge.score(query="What is 2+2?", response="")
    assert res.verdict == "fail"
    assert res.score == 0.0
    assert "empty" in res.feedback.lower()


def test_judge_invalid_thresholds():
    engine = RuleEngine()
    with pytest.raises(ConfigError):
        Judge(engine=engine, pass_threshold=0.4, fail_threshold=0.5)


def test_judge_compare():
    engine = RuleEngine()
    judge = Judge(engine=engine)
    res_a, res_b, winner = judge.compare(
        query="Explain gravity",
        response_a="Gravity is a fundamental force attracting masses.",
        response_b="Gravity is when stuff falls down.",
    )
    assert winner in ("a", "b", "tie")
    assert res_a.score > 0.0
    assert res_b.score > 0.0
