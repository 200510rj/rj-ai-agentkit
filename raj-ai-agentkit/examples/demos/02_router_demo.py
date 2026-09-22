"""
Demo 2 — Router Component
Demonstrates intent classification and route selection using Router and RuleEngine.
"""

from agentkit import Router, RuleEngine, KeywordRule, RouteConfig


def main():
    print("==================================================")
    print("DEMO 2 — ROUTER COMPONENT")
    print("==================================================\n")

    # 1. Define routes
    routes = {
        "general": RouteConfig(name="general", description="General programming and general knowledge questions"),
        "technical": RouteConfig(name="technical", description="Technical issues, code bugs, and database errors"),
        "billing": RouteConfig(name="billing", description="Billing, charges, invoices, and payment refunds"),
    }

    # 2. RuleEngine mapped to route names
    engine = RuleEngine([
        KeywordRule(name="r_general", keywords=["python", "programming", "what is"], verdict="general", confidence=0.92),
        KeywordRule(name="r_tech", keywords=["database", "connection", "error", "bug", "crash"], verdict="technical", confidence=0.95),
        KeywordRule(name="r_billing", keywords=["charged", "invoice", "refund", "payment", "credit card"], verdict="billing", confidence=0.98),
    ])

    # 3. Instantiate Router
    router = Router(
        engine=engine,
        routes=routes,
        fallback="general",
        confidence_threshold=0.3,
    )

    # Test queries
    test_queries = [
        "What is Python?",
        "My application has a database connection error.",
        "I was charged twice.",
    ]

    for query in test_queries:
        result = router.route(query)
        print(f"QUERY          : '{query}'")
        print(f"SELECTED ROUTE : {result.verdict}")
        print(f"CONFIDENCE     : {result.confidence:.2f}")
        print(f"REASONING      : {result.reasoning}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
