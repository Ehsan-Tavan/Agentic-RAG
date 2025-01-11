from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from .base import AbstractVectorStore


class MilvusVectorStore(AbstractVectorStore):
    def __init__(
            self,
            embedding_function: Embeddings,
            connection_args: dict,
            collection_name: str
    ):
        self.collection_name = collection_name
        self.connection_args = connection_args
        self.embedding_function = embedding_function
        self.vector_store = Milvus(
            embedding_function=self.embedding_function,
            connection_args=self.connection_args,
            collection_name=self.collection_name,
        )

    def as_retriever(
            self,
            search_type: str,
            search_kwargs: dict
    ) -> VectorStore:
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
