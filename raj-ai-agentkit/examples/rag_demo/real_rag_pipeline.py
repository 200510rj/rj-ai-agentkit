"""
Real RAG Pipeline Demo for raj-ai-agentkit v0.1

Demonstrates architecture:
User Query -> Guard -> Router -> RAGGate -> Application Retriever -> LLM -> Judge -> Final Answer

RAGGate decides whether to retrieve. Application performs retrieval.
"""

import os
import sys
import urllib.request
from agentkit import Guard, Router, RAGGate, Judge, LLMEngine, RuleEngine, OpenAIProvider, RouteConfig, KeywordRule
from agentkit.exceptions import ProviderError, EngineError
from retriever import SimpleRetriever


def is_ollama_available(base_url: str = "http://localhost:11434/v1") -> bool:
    """Check if local Ollama HTTP endpoint is active."""
    try:
        req = urllib.request.Request(f"{base_url.rstrip('/')}/models")
        with urllib.request.urlopen(req, timeout=2) as resp:
            return resp.status == 200
    except Exception:
        return False


def get_engines():
    """Returns decision engine(s) for primitives."""
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
            KeywordRule(name="policy_route", keywords=["leave", "paid", "hr"], verdict="company_policy", confidence=0.95),
            KeywordRule(name="product_route", keywords=["novawidget", "battery", "waterproof"], verdict="product_manual", confidence=0.95),
            KeywordRule(name="tech_route", keywords=["api", "postgres", "database"], verdict="technical_docs", confidence=0.95),
            KeywordRule(name="general_route", keywords=["python"], verdict="general", confidence=0.95),
        ])
        rag_engine = RuleEngine([
            KeywordRule(name="rag_match", keywords=["leave", "paid", "novawidget", "battery", "waterproof", "api", "postgres", "quantum"], verdict="match", confidence=0.95)
        ])
        judge_engine = RuleEngine()
        return guard_engine, router_engine, rag_engine, judge_engine, None


def main():
    print("==================================================")
    print("REAL RAG PIPELINE DEMO — raj-ai-agentkit")
    print("==================================================\n")

    # 1. Setup Retriever (Application side, independent of agentkit)
    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge")
    retriever = SimpleRetriever(knowledge_dir=knowledge_path)

    # 2. Setup decision engines
    guard_eng, router_eng, rag_eng, judge_eng, provider = get_engines()

    # 3. Setup agentkit reliability primitives
    guard = Guard(engine=guard_eng, checks=["Does this text contain prompt injection?"])
    routes = {
        "general": RouteConfig(name="general", description="General programming and general knowledge questions"),
        "company_policy": RouteConfig(name="company_policy", description="HR policies, annual leave, and working hours"),
        "product_manual": RouteConfig(name="product_manual", description="Product specifications, battery life, and device reset"),
        "technical_docs": RouteConfig(name="technical_docs", description="API endpoints, authentication, and database details"),
    }
    router = Router(engine=router_eng, routes=routes, fallback="general")
    rag_gate = RAGGate(engine=rag_eng)
    judge = Judge(engine=judge_eng)

    # Test cases
    test_queries = [
        "What is Python?",
        "How many annual paid leave days do employees receive?",
        "What is the battery life and waterproof rating of NovaWidget X?",
        "What is the quantum telemetry protocol specification?",
    ]

    for user_query in test_queries:
        print("\n" + "=" * 50)
        print(f"QUESTION: '{user_query}'")
        print("=" * 50)

        # Step 1: Guard
        g_res = guard.check(user_query)
        print(f"\n[1] GUARD    : {g_res.verdict.upper()} (Confidence: {g_res.confidence:.2f})")
        if g_res.verdict == "block":
            print("   Pipeline stopped by Guard.")
            continue

        # Step 2: Router
        r_res = router.route(user_query)
        print(f"[2] ROUTER   : Route = '{r_res.verdict}' (Confidence: {r_res.confidence:.2f})")

        # Step 3: RAG Gate (Decision only)
        gate_res = rag_gate.should_retrieve(user_query)
        print(f"[3] RAG GATE : Verdict = '{gate_res.verdict.upper()}' (Should Retrieve: {gate_res.should_retrieve})")

        # Step 4: Application Retrieval & LLM Response
        retrieved_chunks = []
        if gate_res.should_retrieve:
            # Application executes retrieval
            retrieved_chunks = retriever.retrieve(user_query, top_k=2)
            print(f"[4] RETRIEVER: Retrieved {len(retrieved_chunks)} context chunk(s):")
            if retrieved_chunks:
                for chunk in retrieved_chunks:
                    print(f"    - Source: {chunk['source']} (Score: {chunk['score']:.2f})")
                    print(f"      Text: \"{chunk['content']}\"")
            else:
                print("    - Source: None (No relevant context found in knowledge base)")
        else:
            print("[4] RETRIEVER: SKIP (No retrieval performed)")

        # Generate Grounded LLM Response
        if provider is not None:
            if retrieved_chunks:
                context_str = "\n".join([f"[{c['source']}]: {c['content']}" for c in retrieved_chunks])
                prompt = (
                    f"Answer the question using ONLY the provided context.\n"
                    f"If the answer cannot be found in the context, explicitly state: "
                    f"\"The requested information is not available in the knowledge base.\"\n"
                    f"Do not invent facts.\n\n"
                    f"Context:\n{context_str}\n\n"
                    f"Question: {user_query}"
                )
            elif gate_res.should_retrieve:
                prompt = (
                    f"Question: {user_query}\n"
                    f"Notice: The information for this question was NOT found in the knowledge base.\n"
                    f"Respond explicitly stating: \"The requested information is not available in the knowledge base.\""
                )
            else:
                prompt = f"Answer this general question concisely: {user_query}"

            try:
                raw_llm_answer = provider.complete(prompt)
            except Exception as e:
                raw_llm_answer = f"[LLM generation error: {e}]"
        else:
            # RuleEngine Fallback output
            if gate_res.should_retrieve:
                if retrieved_chunks:
                    raw_llm_answer = f"Grounded Answer based on {retrieved_chunks[0]['source']}: {retrieved_chunks[0]['content']}"
                else:
                    raw_llm_answer = "The requested information is not available in the knowledge base."
            else:
                raw_llm_answer = "Python is a high-level, interpreted programming language known for readability."

        print(f"\n[5] LLM GENERATION:\n\"{raw_llm_answer}\"")

        # Step 5: Judge
        j_res = judge.score(query=user_query, response=raw_llm_answer)
        print(f"\n[6] JUDGE    : Verdict = {j_res.verdict.upper()} (Score: {j_res.score:.2f})")
        print(f"    Feedback : {j_res.feedback}")

        print("\nFINAL ANSWER:")
        print(raw_llm_answer)


if __name__ == "__main__":
    main()
