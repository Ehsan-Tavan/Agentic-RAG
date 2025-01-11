from abc import ABC, abstractmethod
from langchain_core.vectorstores import VectorStore


class AbstractVectorStore(ABC):
    @abstractmethod
    def as_retriever(
            self,
            search_type: str,
            search_kwargs: dict
    ) -> VectorStore:
        """Convert vector store to a retriever."""
