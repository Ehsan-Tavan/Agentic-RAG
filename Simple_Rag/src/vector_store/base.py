from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore


class AbstractVectorStore(ABC):
    """
    Abstract base class for a vector store.

    This class provides an interface for converting a vector store
    into a retriever, which can be used for search or retrieval tasks.
    """
    @abstractmethod
    def as_retriever(
            self,
            search_type: str,
            search_kwargs: dict
    ) -> VectorStore:
        """
        Convert the vector store into a retriever.

        Args:
            search_type: The type of search to perform (e.g., "similarity", ).
            search_kwargs: Additional arguments for the search.

        Returns:
            A retriever instance derived from the vector store.
        """