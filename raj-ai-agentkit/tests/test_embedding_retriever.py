"""
Unit tests for EmbeddingRetriever and semantic RAG pipeline architecture boundaries.
Does NOT require Ollama or a vector database.
"""

import os
from unittest.mock import MagicMock
from agentkit import Guard, Router, RAGGate, Judge, RuleEngine, KeywordRule
from examples.rag_demo.embedding_retriever import EmbeddingRetriever


def get_sample_knowledge_dir():
    return os.path.join(os.path.dirname(__file__), "..", "examples", "rag_demo", "knowledge")


def test_embedding_generation_and_dimension():
    retriever = EmbeddingRetriever(knowledge_dir=get_sample_knowledge_dir())
    vec = retriever.get_embedding("Test embedding text")
    assert isinstance(vec, list)
    assert len(vec) > 0


def test_cosine_similarity():
    vec_a = [1.0, 0.0, 0.0]
    vec_b = [1.0, 0.0, 0.0]
    vec_c = [0.0, 1.0, 0.0]

    sim_identical = EmbeddingRetriever.cosine_similarity(vec_a, vec_b)
    sim_orthogonal = EmbeddingRetriever.cosine_similarity(vec_a, vec_c)

    assert abs(sim_identical - 1.0) < 1e-5
    assert abs(sim_orthogonal - 0.0) < 1e-5


def test_paraphrased_retrieval_success():
    retriever = EmbeddingRetriever(knowledge_dir=get_sample_knowledge_dir())
    chunks = retriever.retrieve("How much vacation time can a staff member take each year?", top_k=1, similarity_threshold=0.50)
    assert len(chunks) > 0
    assert chunks[0]["source"] == "company_policy.txt"
    assert "18 annual paid leave days" in chunks[0]["content"]


def test_unknown_query_returns_no_results():
    retriever = EmbeddingRetriever(knowledge_dir=get_sample_knowledge_dir())
    chunks = retriever.retrieve("Mars colony relocation policy specification", top_k=1, similarity_threshold=0.85)
    assert len(chunks) == 0


def test_retriever_top_k_limit():
    retriever = EmbeddingRetriever(knowledge_dir=get_sample_knowledge_dir())
    chunks = retriever.retrieve("leave days battery endpoint", top_k=1, similarity_threshold=0.01)
    assert len(chunks) <= 1


def test_architecture_gate_skip_does_not_call_embedding_retriever():
    """Verify architecture boundary: RAGGate=SKIP -> EmbeddingRetriever is NOT called."""
    mock_retriever = MagicMock()
    engine = RuleEngine([KeywordRule(name="skip", keywords=["python"], verdict="skip")])
    rag_gate = RAGGate(engine=engine)

    query = "What is Python?"
    gate_res = rag_gate.should_retrieve(query)

    assert gate_res.should_retrieve is False
    assert gate_res.verdict == "skip"

    if gate_res.should_retrieve:
        mock_retriever.retrieve(query)

    mock_retriever.retrieve.assert_not_called()


def test_architecture_gate_retrieve_calls_embedding_retriever():
    """Verify architecture boundary: RAGGate=RETRIEVE -> Application EmbeddingRetriever IS called."""
    mock_retriever = MagicMock()
    mock_retriever.retrieve.return_value = [{"source": "company_policy.txt", "content": "18 days leave", "similarity": 0.85}]

    engine = RuleEngine([KeywordRule(name="match", keywords=["vacation", "leave"], verdict="match")])
    rag_gate = RAGGate(engine=engine)

    query = "How much vacation time?"
    gate_res = rag_gate.should_retrieve(query)

    assert gate_res.should_retrieve is True
    assert gate_res.verdict == "retrieve"

    if gate_res.should_retrieve:
        results = mock_retriever.retrieve(query)

    mock_retriever.retrieve.assert_called_once_with(query)
    assert len(results) == 1
    assert results[0]["similarity"] == 0.85
