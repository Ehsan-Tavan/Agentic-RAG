from typing import Callable
from langchain_core.vectorstores import VectorStore
from Simple_Rag.src.embeddings import HuggingFaceEmbeddingModel
from Simple_Rag.src.vector_store import MilvusVectorStore


def create_retriever_chain(retriever_config: dict) -> Callable[[], VectorStore]:
    """
    Creates a retriever chain using a HuggingFace embedding model and Milvus vector store.

    Args:
        retriever_config: Configuration for the retriever. Expected keys include:
            - "embedding_model" (dict): Embedding model configuration, with:
                - "model_name" (str): Name of the HuggingFace model.
                - "model_kwargs" (dict): Additional kwargs for the embedding model.
            - "vector_db" (dict): Vector database configuration, with:
                - "connection_args" (dict): Arguments for connecting to the Milvus database.
                - "collection_name" (str): Name of the collection in the vector store.
                - "search_type" (str): Search type for the retriever (e.g., "similarity").
                - "search_kwargs" (dict): Additional kwargs for the search.

    Returns:
        A callable that returns a retriever when invoked.
    """
    retriever_model = HuggingFaceEmbeddingModel(
        model_name=retriever_config["embedding_model"]["model_name"],
        model_kwargs=retriever_config["embedding_model"]["model_kwargs"]
    ).model

    def _get_retriever_chain() -> VectorStore:
        """
        Initializes and returns a retriever using the specified configuration.

        Returns:
            An instance of the retriever.
        """
        vector_store = MilvusVectorStore(
            embedding_function=retriever_model,
            connection_args=retriever_config["vector_db"]["connection_args"],
            collection_name=retriever_config["vector_db"]["collection_name"],
        )

        retriever = vector_store.as_retriever(
            search_type=retriever_config["vector_db"]["search_type"],
            search_kwargs=retriever_config["vector_db"]["search_kwargs"]
        )
        return retriever

    return _get_retriever_chain
