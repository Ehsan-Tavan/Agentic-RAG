from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore
from langchain_milvus import Milvus
from .base import AbstractVectorStore


class MilvusVectorStore(AbstractVectorStore):
    """
    A wrapper for Milvus vector store to integrate with LangChain.
    """

    def __init__(
            self,
            embedding_function: Embeddings,
            connection_args: dict,
            collection_name: str
    ):
        """
        Initialize the MilvusVectorStore instance.

        Args:
            embedding_function: An instance of an embedding function
                from LangChain to convert text or data into vector representations.
            connection_args: Connection configuration for Milvus, including
                host, port, and authentication details.
            collection_name: The name of the Milvus collection to use or create.
        """
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
        """
        Return a retriever instance for performing searches in the vector store.

        Args:
            search_type: The type of search to perform, e.g., "similarity" or "semantic".
            search_kwargs: Additional search parameters, such as `k` for the
                number of results to return or filtering options.

        Returns:
            A retriever object compatible with LangChain's interfaces,
            enabling search operations within the vector store.
        """
        return self.vector_store.as_retriever(
            search_type=search_type,
            search_kwargs=search_kwargs,
        )
