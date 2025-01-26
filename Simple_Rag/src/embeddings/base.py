from abc import ABC, abstractmethod
from typing import List


class EmbeddingModel(ABC):
    """
    Abstract base class for embedding models.

    This class defines the required methods for any embedding model implementation.
    """

    @abstractmethod
    def get_dimension(self) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            The dimension of the embedding vector.
        """
        pass

    @abstractmethod
    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """
        Embed a list of documents into vectors.

        Args:
            documents: A list of textual documents to be embedded.

        Returns:
            A list of vectors, where each vector represents a document.
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query into a vector.

        Args:
            text: The input query text to be embedded.

        Returns:
           A vector representing the embedded query.
        """
        pass
