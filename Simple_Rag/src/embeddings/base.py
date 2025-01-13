from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """Abstract base class for embedding models."""

    @abstractmethod
    def get_dimension(self) -> int:
        """Return the dimension of the embedding vector."""
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents into vectors."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Embed a query into vectors."""
        pass
