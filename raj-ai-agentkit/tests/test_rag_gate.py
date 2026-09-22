"""
Unit tests for RAGGate component.
"""

import pytest
from agentkit.components.rag_gate import RAGGate
from agentkit.engines.rule_engine import RuleEngine, KeywordRule


def test_rag_gate_should_retrieve_true():
    engine = RuleEngine(rules=[
        KeywordRule(name="fact_check", keywords=["policy", "document", "spec"], verdict="match")
    ])
    gate = RAGGate(engine)
    res = gate.should_retrieve("What is our company's refund policy?")
    assert res.should_retrieve is True
    assert res.verdict == "retrieve"


def test_rag_gate_should_retrieve_false():
    engine = RuleEngine.from_keywords(block=["policy"])
    gate = RAGGate(engine)
    res = gate.should_retrieve("Hi, how are you today?")
    assert res.should_retrieve is False
    assert res.verdict == "skip"


def test_rag_gate_empty_query():
    engine = RuleEngine()
    gate = RAGGate(engine)
    res = gate.should_retrieve("")
    assert res.should_retrieve is False
    assert res.verdict == "skip"
    assert res.confidence == 1.0


def test_rag_gate_callable_shorthand():
    engine = RuleEngine(rules=[
        KeywordRule(name="doc", keywords=["manual"], verdict="match")
    ])
    gate = RAGGate(engine)
    res = gate("Find manual details")
    assert res.should_retrieve is True
