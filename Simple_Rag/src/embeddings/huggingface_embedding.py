from typing import List
from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from .base import EmbeddingModel


class HuggingFaceEmbeddingModel(EmbeddingModel):
    """
    A concrete implementation of the EmbeddingModel interface using HuggingFace embeddings.
    """
    def __init__(
            self,
            model_name: str,
            model_kwargs: dict
    ):
        """
        Initialize the HuggingFaceEmbeddingModel.

        Args:
            model_name: The name of the HuggingFace model to use for embeddings.
            model_kwargs: Additional keyword arguments to configure the model.
        """
        self.model: Embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)

    def get_dimension(
            self
    ) -> int:
        """
        Get the dimensionality of the embedding vectors.

        Returns:
            The dimension of the sentence embedding vector.
        """
        return self.model._client.get_sentence_embedding_dimension()

    def embed_documents(
            self,
            documents: List[str]
    ) -> List[List[float]]:
        """
        Embed a list of text documents into vectors.

        Args:
            documents: A list of textual documents to embed.

        Returns:
            A list of embedding vectors for the documents.
        """
        return self.model.embed_documents(documents)

    def embed_query(
            self,
            text: str
    ) -> List[float]:
        """
        Embed a single query into a vector.

        Args:
            text: The input query text to embed.

        Returns:
            The embedding vector representing the query.
        """
        return self.model.embed_query(text)
