"""
Demo 3 — RAG Gate Component
Demonstrates retrieval gating decisions (whether external knowledge is required) using RAGGate and RuleEngine.
Note: RAGGate makes the decision ONLY; it does not connect to a vector database.
"""

from agentkit import RAGGate, RuleEngine, KeywordRule


def main():
    print("==================================================")
    print("DEMO 3 — RAG GATE COMPONENT")
    print("==================================================\n")

    # 1. RuleEngine that identifies queries requiring private/external knowledge
    engine = RuleEngine([
        KeywordRule(
            name="private_knowledge",
            keywords=["policy", "leave", "internal", "documentation", "company", "employee"],
            verdict="match",
            confidence=0.95,
        )
    ])

    # 2. Instantiate RAGGate
    rag_gate = RAGGate(engine=engine, threshold=0.5)

    # Test questions
    questions = [
        "What is 2 + 2?",
        "What is our company's leave policy?",
        "What is the latest information in our internal documentation?",
    ]

    for query in questions:
        result = rag_gate.should_retrieve(query)
        retrieve_str = "YES" if result.should_retrieve else "NO"

        print(f"QUERY          : '{query}'")
        print(f"RETRIEVE?      : {retrieve_str} (Verdict: '{result.verdict}')")
        print(f"CONFIDENCE     : {result.confidence:.2f}")
        print(f"REASONING      : {result.reasoning}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
