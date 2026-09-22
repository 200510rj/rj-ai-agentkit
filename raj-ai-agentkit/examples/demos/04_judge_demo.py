"""
Demo 4 — Judge Component
Demonstrates output quality evaluation (scoring, pass/fail/borderline verdict, and feedback generation) using Judge and RuleEngine.
"""

from agentkit import Judge, RuleEngine, CallableRule


def main():
    print("==================================================")
    print("DEMO 4 — JUDGE COMPONENT")
    print("==================================================\n")

    # 1. Custom RuleEngine that evaluates response quality heuristically
    def evaluate_quality(text: str) -> bool:
        return "good" in text.lower() or "python is a high-level" in text.lower()

    engine = RuleEngine([
        CallableRule(name="quality_check", predicate=evaluate_quality, confidence=0.9)
    ])

    # 2. Instantiate Judge with pass/fail thresholds
    judge = Judge(
        engine=engine,
        criteria=["relevance", "accuracy", "completeness", "clarity"],
        pass_threshold=0.7,
        fail_threshold=0.4,
    )

    query = "What is Python?"

    responses = [
        ("Good Answer", "Python is a high-level, interpreted programming language known for readability."),
        ("Bad Answer", ""),  # Empty response
        ("Borderline Answer", "Python is a snake, or maybe some software thing."),
    ]

    for label, response_text in responses:
        print(f"[{label.upper()}]")
        print(f"QUERY    : '{query}'")
        print(f"RESPONSE : '{response_text}'")

        result = judge.score(query=query, response=response_text)

        print("[JUDGE EVALUATION]")
        print(f"Score    : {result.score:.2f}")
        print(f"Verdict  : {result.verdict.upper()}")
        print(f"Feedback : {result.feedback}")
        print("-" * 50 + "\n")


if __name__ == "__main__":
    main()
