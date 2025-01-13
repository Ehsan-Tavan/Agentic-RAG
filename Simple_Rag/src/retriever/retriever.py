from Simple_Rag.src.embeddings import HuggingFaceEmbeddingModel
from Simple_Rag.src.vector_store import MilvusVectorStore


def create_retriever_chain(retriever_config):
    retriever_model = HuggingFaceEmbeddingModel(
        model_name=retriever_config["embedding_model"]["model_name"],
        model_kwargs=retriever_config["embedding_model"]["model_kwargs"]
    ).model

    def _get_retriever_chain():
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
