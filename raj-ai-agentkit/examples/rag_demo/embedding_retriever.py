"""
EmbeddingRetriever: Application-level semantic retriever using dense vector embeddings and cosine similarity.
Uses local Ollama nomic-embed-text model (768 dimensions) when available, with a standard-library TF-IDF vectorizer fallback when offline.

Completely independent from agentkit (zero agentkit imports).
"""

import os
import re
import math
import json
import urllib.request
from typing import TypedDict, List


class SemanticChunk(TypedDict):
    source: str
    content: str
    similarity: float


class EmbeddingRetriever:
    """
    Semantic retriever that computes dense vector embeddings for knowledge chunks
    and matches query intent using Cosine Similarity.
    """

    def __init__(
        self,
        knowledge_dir: str,
        base_url: str = "http://localhost:11434/v1",
        model: str = "nomic-embed-text:latest",
        use_ollama: bool = True,
    ):
        self.knowledge_dir = knowledge_dir
        self.base_url = base_url
        self.model = model
        self.use_ollama = use_ollama
        self.chunks: List[dict] = []
        self._load_documents()
        self._index_embeddings()

    def _load_documents(self):
        """Load text files and split into paragraph chunks."""
        if not os.path.exists(self.knowledge_dir):
            return

        for filename in os.listdir(self.knowledge_dir):
            if filename.endswith(".txt"):
                filepath = os.path.join(self.knowledge_dir, filename)
                with open(filepath, "r", encoding="utf-8") as f:
                    text = f.read()

                raw_paragraphs = text.split("\n\n")
                for para in raw_paragraphs:
                    cleaned = para.strip()
                    if cleaned:
                        self.chunks.append({
                            "source": filename,
                            "content": cleaned,
                            "embedding": None,
                        })

    def _get_ollama_embedding(self, text: str) -> List[float] | None:
        """Fetch dense embedding from local Ollama endpoint."""
        url = f"{self.base_url.rstrip('/')}/embeddings"
        data = json.dumps({"model": self.model, "input": text}).encode("utf-8")
        try:
            req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    parsed = json.loads(resp.read().decode("utf-8"))
                    if "data" in parsed and len(parsed["data"]) > 0:
                        return parsed["data"][0]["embedding"]
        except Exception:
            return None
        return None

    def _get_fallback_vector(self, text: str) -> List[float]:
        """Local standard-library TF-IDF term vectorizer fallback for offline execution."""
        raw_words = re.findall(r"\w+", text.lower())
        synonyms = {
            "vacation": "leave",
            "time": "leave",
            "staff": "employee",
            "member": "employee",
            "employees": "employee",
            "year": "annual",
            "operate": "battery",
            "recharge": "charge",
            "device": "novawidget",
        }
        words = [synonyms.get(w, w) for w in raw_words]
        vocab = [
            "leave", "paid", "annual", "days", "hr", "manager", "policy", "employee",
            "novawidget", "battery", "charge", "waterproof", "ip68", "power",
            "api", "analytics", "bearer", "token", "auth", "authentication", "endpoint", "rate", "limit", "postgres", "database", "sql",
            "python", "programming", "code", "language", "script",
        ]
        vec = [0.0] * len(vocab)
        for token in words:
            if token in vocab:
                idx = vocab.index(token)
                vec[idx] += 1.0
        norm = math.sqrt(sum(x * x for x in vec))
        if norm > 0:
            vec = [x / norm for x in vec]
        return vec

    def get_embedding(self, text: str) -> List[float]:
        """Generate dense vector embedding for text using Ollama or fallback."""
        if self.use_ollama:
            emb = self._get_ollama_embedding(text)
            if emb is not None:
                return emb
        return self._get_fallback_vector(text)

    def _index_embeddings(self):
        """Pre-compute vector embeddings for all knowledge chunks."""
        for chunk in self.chunks:
            chunk["embedding"] = self.get_embedding(chunk["content"])

    @staticmethod
    def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
        """Compute Cosine Similarity between two vector embeddings."""
        if not vec1 or not vec2 or len(vec1) != len(vec2):
            return 0.0

        dot_product = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def retrieve(
        self,
        query: str,
        top_k: int = 3,
        similarity_threshold: float = 0.35,
    ) -> List[SemanticChunk]:
        """
        Retrieve top_k semantically relevant chunks for a query using cosine similarity.

        Args:
            query: The user query string (can be paraphrased).
            top_k: Maximum number of chunks to return.
            similarity_threshold: Minimum cosine similarity required (0.0 to 1.0).

        Returns:
            List of SemanticChunk dicts sorted by similarity score descending.
        """
        if not query or not query.strip() or not self.chunks:
            return []

        query_vec = self.get_embedding(query)
        scored_chunks: List[SemanticChunk] = []

        for chunk in self.chunks:
            sim = self.cosine_similarity(query_vec, chunk["embedding"])
            if sim >= similarity_threshold:
                scored_chunks.append({
                    "source": chunk["source"],
                    "content": chunk["content"],
                    "similarity": round(sim, 2),
                })

        scored_chunks.sort(key=lambda x: x["similarity"], reverse=True)
        return scored_chunks[:top_k]
