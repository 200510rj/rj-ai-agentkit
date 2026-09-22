"""
Abstract Base Classes for LLM providers and Vector Stores.
"""

from abc import ABC, abstractmethod
from typing import Type, TypeVar
from pydantic import BaseModel
from agentkit.types import SearchResult

T = TypeVar("T", bound=BaseModel)


class LLMProvider(ABC):
    """Abstraction over any LLM API (OpenAI, Ollama, Anthropic, Gemini, etc.)."""

    @abstractmethod
    def complete(self, prompt: str, **kwargs) -> str:
        """
        Generate raw text completion for a prompt.

        Args:
            prompt: Text prompt to complete.
            **kwargs: Provider-specific inference parameters (temperature, max_tokens, etc.).

        Returns:
            String response from the LLM.
        """
        pass

    @abstractmethod
    def structured_output(self, prompt: str, schema: Type[T], **kwargs) -> T:
        """
        Generate response conforming strictly to a Pydantic schema.

        Args:
            prompt: Text prompt.
            schema: Pydantic model class to validate and parse output into.
            **kwargs: Provider-specific parameters.

        Returns:
            An instance of the schema class.
        """
        pass


class VectorStore(ABC):
    """Abstraction over any vector database (Chroma, Pinecone, FAISS, In-Memory, etc.)."""

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchResult]:
        """
        Search vector store for top_k relevant documents.

        Args:
            query: Query string.
            top_k: Maximum number of results to return.

        Returns:
            List of SearchResult objects.
        """
        pass

    @abstractmethod
    def add(self, documents: list[str], metadatas: list[dict] | None = None) -> None:
        """
        Add documents to the vector store.

        Args:
            documents: List of text documents.
            metadatas: Optional list of metadata dicts corresponding to documents.
        """
        pass
