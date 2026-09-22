"""
Retriever Comparison Demo: SimpleRetriever (Keyword) vs. EmbeddingRetriever (Semantic)
Demonstrates why semantic vector retrieval handles paraphrasing and synonym matching where keyword matching fails.
"""

import os
from retriever import SimpleRetriever
from embedding_retriever import EmbeddingRetriever


def main():
    print("==================================================")
    print("RETRIEVER COMPARISON — KEYWORD VS. EMBEDDING (SEMANTIC)")
    print("==================================================\n")

    knowledge_path = os.path.join(os.path.dirname(__file__), "knowledge")

    keyword_retriever = SimpleRetriever(knowledge_dir=knowledge_path)
    embedding_retriever = EmbeddingRetriever(knowledge_dir=knowledge_path)

    test_queries = [
        ("Test 1 — Leave Paraphrase", "How much vacation time can a staff member take each year?"),
        ("Test 2 — Product Semantics", "How long can the device operate before needing a recharge?"),
        ("Test 3 — Technical Semantics", "Which API endpoint is used for analytics and what authentication does it require?"),
        ("Test 4 — Unknown Information", "What is the company's Mars colony relocation policy?"),
    ]

    for label, query in test_queries:
        print("=" * 60)
        print(f"QUERY [{label}]:\n'{query}'")
        print("=" * 60)

        # Keyword Retriever
        kw_results = keyword_retriever.retrieve(query, top_k=1, score_threshold=0.35)
        print("\n[KEYWORD RETRIEVER (SimpleRetriever)]:")
        if kw_results:
            print(f"  - Document   : {kw_results[0]['source']}")
            print(f"  - Overlap    : {kw_results[0]['score']:.2f}")
            print(f"  - Snippet    : \"{kw_results[0]['content'][:90]}...\"")
        else:
            print("  - Document   : NONE (Keyword overlap below threshold)")

        # Embedding Retriever
        emb_results = embedding_retriever.retrieve(query, top_k=1, similarity_threshold=0.60)
        print("\n[EMBEDDING RETRIEVER (EmbeddingRetriever)]:")
        if emb_results:
            print(f"  - Document   : {emb_results[0]['source']}")
            print(f"  - Similarity : {emb_results[0]['similarity']:.2f}")
            print(f"  - Snippet    : \"{emb_results[0]['content'][:90]}...\"")
        else:
            print("  - Document   : NONE (Cosine similarity below threshold)")

        print("\n")


if __name__ == "__main__":
    main()
