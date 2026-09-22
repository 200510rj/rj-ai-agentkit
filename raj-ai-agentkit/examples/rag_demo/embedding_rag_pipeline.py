"""
Embedding RAG Pipeline Demo for raj-ai-agentkit v0.1

Demonstrates complete architecture with a semantic embedding retriever:
User Query -> Guard -> Router -> RAGGate -> EmbeddingRetriever -> LLM -> Judge -> Final Answer

RAGGate decides whether retrieval is required. EmbeddingRetriever selects relevant document chunks using cosine similarity.
"""

import os
import sys
import urllib.request
from agentkit import Guard, Router, RAGGate, Judge, LLMEngine, RuleEngine, OpenAIProvider, RouteConfig, KeywordRule
from agentkit.exceptions import ProviderError, EngineError
from embedding_retriever import EmbeddingRetriever


def is_ollama_available(base_url: str = "http://localhost:11434/v1") -> bool:
    """Check if local Ollama HTTP endpoint is active."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_engines():
    """Returns decision engines for primitives."""
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model_name = os.environ.get("AGENTKIT_OLLAMA_MODEL", "qwen2.5-coder:7b")

    if is_ollama_available(base_url):
        print(f"[ENGINE]: Using Ollama Local LLM ({model_name} at {base_url})")
        provider = OpenAIProvider(model=model_name, base_url=base_url)
        engine = LLMEngine(provider=provider)
        return engine, engine, engine, engine, provider
    else:
        print("[ENGINE]: Ollama not active. Falling back to zero-dependency RuleEngine for demo.")
        guard_engine = RuleEngine([
            KeywordRule(name="inj", keywords=["ignore previous", "system prompt"], verdict="block", confidence=0.99)
        ])
        router_engine = RuleEngine([
            KeywordRule(name="policy_route", keywords=["vacation", "leave", "paid", "hr", "staff"], verdict="company_policy", confidence=0.95),
            KeywordRule(name="product_route", keywords=["novawidget", "device", "operate", "recharge", "battery", "waterproof"], verdict="product_manual", confidence=0.95),
            KeywordRule(name="tech_route", keywords=["api", "analytics", "bearer", "authentication", "postgres", "database"], verdict="technical_docs", confidence=0.95),
            KeywordRule(name="general_route", keywords=["python"], verdict="general", confidence=0.95),
        ])
        rag_engine = RuleEngine([
            KeywordRule(name="rag_match", keywords=["vacation", "leave", "paid", "device", "recharge", "operate", "battery", "api", "analytics", "mars"], verdict="match", confidence=0.95)
        ])
        judge_engine = RuleEngine()
        return guard_engine, router_engine, rag_engine, judge_engine, None


def main():
    print("==================================================")
    print("EMBEDDING RAG PIPELINE DEMO — raj-ai-agentkit")
    print("==================================================\n")

    # 1. Setup Application EmbeddingRetriever
    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge")
    retriever = EmbeddingRetriever(knowledge_dir=knowledge_path)

    # 2. Setup agentkit decision engines
    guard_eng, router_eng, rag_eng, judge_eng, provider = get_engines()

    # 3. Setup agentkit reliability primitives
    guard = Guard(engine=guard_eng, checks=["Does this text contain prompt injection?"])
    routes = {
        "general": RouteConfig(name="general", description="General programming and general knowledge questions"),
        "company_policy": RouteConfig(name="company_policy", description="HR policies, annual leave, vacation time, and working hours"),
        "product_manual": RouteConfig(name="product_manual", description="Product specifications, battery operation, and charging"),
        "technical_docs": RouteConfig(name="technical_docs", description="API endpoints, authentication, and database details"),
    }
    router = Router(engine=router_eng, routes=routes, fallback="general")
    rag_gate = RAGGate(engine=rag_eng)
    judge = Judge(engine=judge_eng)

    test_queries = [
        "What is Python?",
        "How much vacation time can a staff member take each year?",
        "How long can the device operate before needing a recharge?",
        "Which API endpoint is used for analytics and what authentication does it require?",
        "What is the company's Mars colony relocation policy?",
    ]

    for user_query in test_queries:
        print("\n" + "=" * 60)
        print(f"QUESTION: '{user_query}'")
        print("=" * 60)

        # Step 1: Guard
        g_res = guard.check(user_query)
        print(f"\n[1] GUARD       : {g_res.verdict.upper()} (Confidence: {g_res.confidence:.2f})")
        if g_res.verdict == "block":
            print("    Pipeline stopped by Guard.")
            continue

        # Step 2: Router
        r_res = router.route(user_query)
        print(f"[2] ROUTER      : Route = '{r_res.verdict}' (Confidence: {r_res.confidence:.2f})")

        # Step 3: RAG Gate (Decision ONLY)
        gate_res = rag_gate.should_retrieve(user_query)
        print(f"[3] RAG GATE    : Verdict = '{gate_res.verdict.upper()}' (Should Retrieve: {gate_res.should_retrieve})")

        # Step 4: Application Embedding Retrieval
        retrieved_chunks = []
        if gate_res.should_retrieve:
            print("[4] RETRIEVER   : EmbeddingRetriever (Cosine Similarity)")
            retrieved_chunks = retriever.retrieve(user_query, top_k=2, similarity_threshold=0.60)
            if retrieved_chunks:
                for c in retrieved_chunks:
                    print(f"    - Document   : {c['source']}")
                    print(f"      Similarity : {c['similarity']:.2f}")
                    print(f"      Context    : \"{c['content'][:100]}...\"")
            else:
                print("    - Document   : NONE (No chunk met similarity threshold)")
        else:
            print("[4] RETRIEVER   : SKIP (Embedding retriever did not run)")

        # Step 5: LLM Generation
        if provider is not None:
            if retrieved_chunks:
                context_str = "\n".join([f"[{c['source']}]: {c['content']}" for c in retrieved_chunks])
                prompt = (
                    f"Answer the user's question using the provided context as the source of truth.\n"
                    f"If the answer cannot be found in the context, say that the information is not available.\n"
                    f"Do not invent facts.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {user_query}"
                )
            elif gate_res.should_retrieve:
                prompt = (
                    f"Question: {user_query}\n"
                    f"Notice: Information was not found in knowledge base.\n"
                    f"Respond: \"The requested information is not available in the knowledge base.\""
                )
            else:
                prompt = f"Answer this general question concisely: {user_query}"

            try:
                raw_llm_answer = provider.complete(prompt)
            except Exception as e:
                raw_llm_answer = f"[LLM generation error: {e}]"
        else:
            if gate_res.should_retrieve:
                if retrieved_chunks:
                    raw_llm_answer = f"Grounded Answer ({retrieved_chunks[0]['source']}): {retrieved_chunks[0]['content']}"
                else:
                    raw_llm_answer = "The requested information is not available in the knowledge base."
            else:
                raw_llm_answer = "Python is a high-level interpreted programming language known for readability."

        print(f"\n[5] LLM ANSWER:\n\"{raw_llm_answer}\"")

        # Step 6: Judge
        j_res = judge.score(query=user_query, response=raw_llm_answer)
        print(f"\n[6] JUDGE       : Verdict = {j_res.verdict.upper()} (Score: {j_res.score:.2f})")
        print(f"    Feedback    : {j_res.feedback}")


if __name__ == "__main__":
    main()
