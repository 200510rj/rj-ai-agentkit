"""
Unit tests for LayaEngine (mocked laya.Router).
"""

from unittest.mock import MagicMock
from agentkit.engines.laya_engine import LayaEngine


def test_laya_engine_check():
    mock_router = MagicMock()
    mock_router.predict.return_value = {
        "answer": {"noul": True},
        "confidence": 0.98,
        "reasoning": "Laya binary decision",
    }
    engine = LayaEngine(router_instance=mock_router)
    res = engine.check("Sample text", instructions="Is valid?")
    assert res.result is True
    assert res.confidence == 0.98


def test_laya_engine_classify():
    mock_router = MagicMock()
    mock_router.predict.return_value = {
        "answer": {"choice": "billing"},
        "confidence": 0.94,
        "scores": {"billing": 0.94, "tech": 0.06},
    }
    engine = LayaEngine(router_instance=mock_router)
    res = engine.classify("Payment issue", options={"billing": "Billing", "tech": "Tech"})
    assert res.selected == "billing"
    assert res.confidence == 0.94


def test_laya_engine_score():
    mock_router = MagicMock()
    mock_router.predict.return_value = {
        "answer": {"score": 0.89},
        "criteria_scores": {"relevance": 0.9, "clarity": 0.88},
        "confidence": 0.95,
    }
    engine = LayaEngine(router_instance=mock_router)
    res = engine.score("Response text", criteria=["relevance", "clarity"])
    assert res.score == 0.89
    assert res.criteria_scores["relevance"] == 0.9
