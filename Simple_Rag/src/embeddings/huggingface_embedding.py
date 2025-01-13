from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from .base import EmbeddingModel


class HuggingFaceEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str, model_kwargs: dict):
        self.model: Embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    def get_dimension(self) -> int:
        return self.model._client.get_sentence_embedding_dimension()

    def embed_documents(
            self,
            documents: List[str]
    ) -> List[List[float]]:
        return self.model.embed_documents(documents)

    def embed_query(
            self,
            text: str
    ) -> List[float]:
        return self.model.embed_query(text)
