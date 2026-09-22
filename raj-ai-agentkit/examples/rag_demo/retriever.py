"""
SimpleRetriever: A simple, deterministic keyword-based retriever for the RAG demo.
Uses Python standard library only. Completely independent from agentkit.
"""

import os
import re
from typing import TypedDict, List


class RetrievedChunk(TypedDict):
    source: str
    content: str
    score: float


class SimpleRetriever:
    """
    Standard-library retriever that loads text documents, splits them into paragraph chunks,
    and scores chunks based on query keyword overlap.
    """

    def __init__(self, knowledge_dir: str):
        """
        Initialize retriever by indexing all .txt files in knowledge_dir.
        """
        self.knowledge_dir = knowledge_dir
        self.chunks: List[dict] = []
        self._load_documents()

    def _load_documents(self):
        """Load text files and split into paragraph chunks."""
        if not os.path.exists(self.knowledge_dir):
            return

        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.knowledge_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                # Split by double newlines into paragraphs/sections
                raw_paragraphs = text.split("\n\n")
                for para in raw_paragraphs:
                    cleaned_para = para.strip()
                    if cleaned_para:
                        self.chunks.append({
                            "source": filename,
                            "content": cleaned_para,
                        })

    def retrieve(self, query: str, top_k: int = 3, score_threshold: float = 0.35) -> List[RetrievedChunk]:
        """
        Retrieve top_k matching chunks for a query.

        Args:
            query: The user query string.
            top_k: Maximum number of chunks to return.
            score_threshold: Minimum score threshold to consider a match relevant.

        Returns:
            List of RetrievedChunk dicts sorted by score descending.
        """
        if not query or not query.strip() or not self.chunks:
            return []

        # Tokenize query into lowercase words (stopwords filtered)
        stopwords = {"what", "is", "our", "the", "a", "an", "and", "or", "in", "on", "at", "to", "for", "of", "with", "by", "how", "many", "do", "does"}
        query_words = [
            w.lower() for w in re.findall(r"\w+", query)
            if w.lower() not in stopwords and len(w) > 1
        ]

        if not query_words:
            return []

        scored_chunks: List[RetrievedChunk] = []
        for chunk in self.chunks:
            chunk_text_lower = chunk["content"].lower()
            matches = sum(1 for word in query_words if word in chunk_text_lower)
            score = matches / len(query_words)

            if score >= score_threshold:
                scored_chunks.append({
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "score": round(score, 2),
                })

        # Sort by score descending
        scored_chunks.sort(key=lambda x: x["score"], reverse=True)
        return scored_chunks[:top_k]
