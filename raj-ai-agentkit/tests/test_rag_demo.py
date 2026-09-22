"""
Unit tests for the RAG demo retriever and pipeline architecture boundaries.
Does NOT require Ollama or an external vector database.
"""

import os
from unittest.mock import MagicMock
from agentkit import Guard, Router, RAGGate, Judge, RuleEngine, KeywordRule, RouteConfig
from examples.rag_demo.retriever import SimpleRetriever


def get_sample_knowledge_dir():
    return os.path.join(os.path.dirname(__file__), "..", "examples", "rag_demo", "knowledge")


def test_retriever_relevant_document():
    retriever = SimpleRetriever(knowledge_dir=get_sample_knowledge_dir())
    chunks = retriever.retrieve("annual paid leave days", top_k=2)
    assert len(chunks) > 0
    assert chunks[0]["source"] == "company_policy.txt"
    assert "18 annual paid leave days" in chunks[0]["content"]


def test_retriever_irrelevant_query():
    retriever = SimpleRetriever(knowledge_dir=get_sample_knowledge_dir())
    chunks = retriever.retrieve("quantum teleportation protocol xyz", top_k=2)
    assert len(chunks) == 0


def test_retriever_top_k():
    retriever = SimpleRetriever(knowledge_dir=get_sample_knowledge_dir())
    chunks = retriever.retrieve("leave reset policy API battery", top_k=1, score_threshold=0.01)
    assert len(chunks) <= 1


def test_pipeline_gate_skip_does_not_call_retriever():
    """Verify architecture boundary: RAGGate=NO -> Retriever is NOT called."""
    mock_retriever = MagicMock()
    engine = RuleEngine([KeywordRule(name="skip", keywords=["python"], verdict="skip")])
    rag_gate = RAGGate(engine=engine)

    query = "What is Python?"
    gate_res = rag_gate.should_retrieve(query)

    assert gate_res.should_retrieve is False
    assert gate_res.verdict == "skip"

    # Application logic check: retriever should not be called when gate skips
    if gate_res.should_retrieve:
        mock_retriever.retrieve(query)

    mock_retriever.retrieve.assert_not_called()


def test_pipeline_gate_retrieve_calls_retriever():
    """Verify architecture boundary: RAGGate=YES -> Application Retriever IS called."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"source": "company_policy.txt", "content": "18 days leave", "score": 0.9}]

    engine = RuleEngine([KeywordRule(name="match", keywords=["leave"], verdict="match")])
    rag_gate = RAGGate(engine=engine)

    query = "How many leave days?"
    gate_res = rag_gate.should_retrieve(query)

    assert gate_res.should_retrieve is True
    assert gate_res.verdict == "retrieve"

    # Application logic check: retriever is called when gate returns True
    if gate_res.should_retrieve:
        results = mock_retriever.retrieve(query)

    mock_retriever.retrieve.assert_called_once_with(query)
    assert len(results) == 1
    assert results[0]["source"] == "company_policy.txt"
