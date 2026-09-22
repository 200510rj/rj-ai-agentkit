"""
raj-ai-agentkit — Full Pipeline Flow (Guard -> Route -> Gate -> Execute -> Judge)
"""

from agentkit import Guard, Router, RAGGate, Judge, RuleEngine, RouteConfig, KeywordRule


def main():
    # Configure decision engine with domain rules
    engine = RuleEngine(rules=[
        KeywordRule(name="injection", keywords=["system prompt", "ignore instructions"], verdict="block"),
        KeywordRule(name="billing_rule", keywords=["payment", "refund", "invoice"], verdict="billing"),
        KeywordRule(name="tech_rule", keywords=["crash", "bug", "error"], verdict="technical"),
        KeywordRule(name="docs_rule", keywords=["policy", "docs", "manual"], verdict="match"),
    ])

    guard = Guard(engine=engine)
    router = Router(
        engine=engine,
        routes={
            "billing": RouteConfig(name="billing", description="Billing and invoice questions"),
            "technical": RouteConfig(name="technical", description="Technical support and errors"),
            "general": RouteConfig(name="general", description="General inquiries"),
        },
        fallback="general",
    )
    rag_gate = RAGGate(engine=engine)
    judge = Judge(engine=engine)

    # Test query
    user_query = "Please check the refund policy for my payment invoice"

    print("=== Agent Reliability Pipeline ===")
    print(f"User Query: '{user_query}'\n")

    # Step 1: Guard check
    guard_res = guard.check(user_query)
    print(f"1. [Guard] Verdict: {guard_res.verdict} (Confidence: {guard_res.confidence:.2f})")
    if guard_res.verdict == "block":
        print("   Blocked by Guard! Terminating pipeline.")
        return

    # Step 2: Intent Routing
    route_res = router.route(user_query)
    print(f"2. [Router] Selected Route: '{route_res.verdict}' (Confidence: {route_res.confidence:.2f})")

    # Step 3: RAG Gate decision
    gate_res = rag_gate.should_retrieve(user_query)
    print(f"3. [RAGGate] Should Retrieve: {gate_res.should_retrieve} (Verdict: {gate_res.verdict})")

    # Step 4: Execution (caller side)
    if gate_res.should_retrieve:
        retrieved_docs = "Refund policy: Full refunds within 30 days of purchase."
        print(f"4. [Execution] Context Retrieved: '{retrieved_docs}'")
        agent_response = f"According to our refund policy: Full refunds are provided within 30 days of purchase."
    else:
        print("4. [Execution] Direct Generation (Retrieval Skipped)")
        agent_response = "I can assist you with your payment."

    print(f"   Agent Output: '{agent_response}'\n")

    # Step 5: Quality Evaluation by Judge
    judge_res = judge.score(query=user_query, response=agent_response)
    print(f"5. [Judge] Verdict: {judge_res.verdict} | Score: {judge_res.score:.2f}")
    print(f"   Feedback: {judge_res.feedback}")


if __name__ == "__main__":
    main()
