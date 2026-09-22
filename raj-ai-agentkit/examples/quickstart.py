"""
raj-ai-agentkit — 5-Line Quickstart Example
"""

from agentkit import Guard, Router, RAGGate, Judge, RuleEngine, RouteConfig

# 1. Zero-dependency Rule Engine
engine = RuleEngine.from_keywords(block=["ignore previous", "system prompt"])

# 2. Safety Guard
guard = Guard(engine=engine)

# 3. Intent Router
router = Router(
    engine=engine,
    routes={
        "billing": RouteConfig(name="billing", description="Invoices, payments, refunds"),
        "tech": RouteConfig(name="tech", description="Bugs, crashes, system errors"),
    },
    fallback="tech",
)

# 4. RAG Gate
rag_gate = RAGGate(engine=engine)

# 5. Quality Judge
judge = Judge(engine=engine)

# Usage Demo
query = "I was charged twice on my credit card invoice"

guard_res = guard(query)
print(f"[Guard] Verdict: {guard_res.verdict}")

if guard_res.verdict == "allow":
    route_res = router(query)
    print(f"[Router] Selected Route: {route_res.verdict}")

    gate_res = rag_gate(query)
    print(f"[RAGGate] Should Retrieve: {gate_res.should_retrieve}")

    simulated_response = "We apologize for the double charge. We have issued a full refund to your account."
    judge_res = judge(query=query, response=simulated_response)
    print(f"[Judge] Verdict: {judge_res.verdict} (Score: {judge_res.score:.2f})")
