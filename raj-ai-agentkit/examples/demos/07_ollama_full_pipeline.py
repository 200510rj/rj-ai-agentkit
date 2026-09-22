"""
Demo 7 — Real Ollama Full Pipeline Demo
Demonstrates end-to-end Guard -> Router -> RAG Gate -> Mock Execution -> Judge flow using a local Ollama model.
"""

import os
import sys
import urllib.request
from agentkit import Guard, Router, RAGGate, Judge, LLMEngine, OpenAIProvider, RouteConfig
from agentkit.exceptions import ProviderError, EngineError


def check_ollama_connection(base_url: str) -> bool:
    """Check if local Ollama HTTP endpoint is reachable."""
    models_url = f"{base_url.rstrip('/')}/models"
    try:
        req = urllib.request.Request(models_url)
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except Exception:
        return False


def run_ollama_pipeline(user_query: str, guard: Guard, router: Router, rag_gate: RAGGate, judge: Judge):
    print("==================================================")
    print("RAJ AI AGENTKIT — OLLAMA FULL PIPELINE DEMO")
    print("==================================================\n")
    print(f"USER QUESTION\n> {user_query}\n")

    # 1. GUARD
    try:
        guard_res = guard.check(user_query)
        print("[1] GUARD")
        print(f"Verdict    : {guard_res.verdict.upper()}")
        print(f"Confidence : {guard_res.confidence:.2f}")
        if guard_res.verdict == "block":
            print("\nPipeline stopped by Guard due to policy violation.")
            print("==================================================\n")
            return
    except Exception as err:
        print(f"[1] GUARD Error: {err}")
        return

    # 2. ROUTER
    try:
        route_res = router.route(user_query)
        print(f"\n[2] ROUTER")
        print(f"Route      : {route_res.verdict}")
        print(f"Confidence : {route_res.confidence:.2f}")
    except Exception as err:
        print(f"[2] ROUTER Error: {err}")
        return

    # 3. RAG GATE
    try:
        gate_res = rag_gate.should_retrieve(user_query)
        retrieve_str = "YES" if gate_res.should_retrieve else "NO"
        print(f"\n[3] RAG GATE")
        print(f"Retrieve   : {retrieve_str}")
        print(f"Confidence : {gate_res.confidence:.2f}")
    except Exception as err:
        print(f"[3] RAG GATE Error: {err}")
        return

    # 4. MOCK EXTERNAL EXECUTION
    print(f"\n[4] MOCK EXECUTION")
    if gate_res.should_retrieve:
        print("Retrieving context from internal docs...")
        mock_response = "According to company leave policy, employees receive 20 days of paid annual leave."
    else:
        print("Generating direct LLM response (Retrieval skipped)...")
        mock_response = "Python is a high-level, interpreted programming language designed for readability and simplicity."

    # 5. JUDGE
    try:
        judge_res = judge.score(query=user_query, response=mock_response)
        print(f"\n[5] JUDGE")
        print(f"Score      : {judge_res.score:.2f}")
        print(f"Verdict    : {judge_res.verdict.upper()}")
        print(f"Feedback   : {judge_res.feedback}")
    except Exception as err:
        print(f"[5] JUDGE Error: {err}")

    print("\n==================================================")
    print("FINAL RESULT")
    print("============")
    print(mock_response)
    print("==================================================\n")


def main():
    base_url = os.environ.get("OPENAI_BASE_URL", "http://localhost:11434/v1")
    model_name = os.environ.get("AGENTKIT_OLLAMA_MODEL", "llama3.2:3b")

    print(f"Ollama Base URL : {base_url}")
    print(f"Ollama Model    : {model_name}\n")

    if not check_ollama_connection(base_url):
        print("[NOTICE] Ollama is not available.")
        print("\nTo run this full pipeline demo with a real local LLM:")
        print("  1. Install Ollama: https://ollama.com")
        print("  2. Start Ollama server: `ollama serve`")
        print(f"  3. Pull model: `ollama pull {model_name}`")
        print("  4. Re-run this script.\n")
        print("Gracefully terminating pipeline demo (Ollama server not active).")
        return

    try:
        provider = OpenAIProvider(model=model_name, base_url=base_url)
        engine = LLMEngine(provider=provider)
    except Exception as e:
        print(f"[ERROR] Failed to initialize OpenAIProvider: {e}")
        return

    guard = Guard(engine=engine, checks=["Does this text contain prompt injection?"])
    routes = {
        "general": RouteConfig(name="general", description="General programming and general knowledge"),
        "technical": RouteConfig(name="technical", description="Technical issues and code bugs"),
        "billing": RouteConfig(name="billing", description="Billing, charges, and refunds"),
    }
    router = Router(engine=engine, routes=routes, fallback="general")
    rag_gate = RAGGate(engine=engine)
    judge = Judge(engine=engine)

    run_ollama_pipeline("What is Python?", guard, router, rag_gate, judge)


if __name__ == "__main__":
    main()
