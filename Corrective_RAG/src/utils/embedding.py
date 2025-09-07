from abc import ABC, abstractmethod
from typing import List, Type, Dict
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings



class EmbeddingModel(ABC):
    def __init__(self, model_name: str = None):
        pass

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Convert a list of documents into embeddings."""
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """Convert a single query into an embedding vector."""
        pass


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        super().__init__(model_name=model_name)
        self.model = SentenceTransformerEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


class OpenAIEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "text-embedding-3-small"):
        super().__init__(model_name=model_name)
        self.model = OpenAIEmbeddings(model=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)

    def embed_query(self, text: str) -> List[float]:
        return self.model.embed_query(text)


class EmbeddingFactory:
    """Factory to create embedding models dynamically."""

    _embedding_map: Dict[str, Type[EmbeddingModel]] = {
        "openai": OpenAIEmbeddingModel,
        "sentence-transformer": SentenceTransformerEmbeddingModel
    }

    @staticmethod
    def create(model_type: str, model_name: str = None) -> EmbeddingModel:
        """
        Create an embedding model instance.

        Args:
            model_type: "openai" or "sentence-transformer"
            model_name: specific model name for the embedding

        Returns:
            An instance of EmbeddingModel
        """
        model_cls = EmbeddingFactory._embedding_map.get(model_type.lower())
        if not model_cls:
            raise ValueError(f"Unknown embedding model type: {model_type}")

        if model_name is None:
            return model_cls()
        return model_cls(model_name=model_name)
