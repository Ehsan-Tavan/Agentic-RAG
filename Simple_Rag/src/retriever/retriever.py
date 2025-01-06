from langchain_huggingface import HuggingFaceEmbeddings
from .helper import get_milvus_collection


def get_embedding_function(retriever_model_config):
    return HuggingFaceEmbeddings(
        model_name=retriever_model_config["model_name"],
        model_kwargs=retriever_model_config["model_kwargs"],
    )


def create_retriever_chain(retriever_model_config):
    retriever_model = get_embedding_function(retriever_model_config)

    def _get_retriever_chain(retriever_config):

        vector_store = get_milvus_collection(
            embedding_function=retriever_model,
            connection_args=retriever_config["retriever_db_config"],
            collection_name=retriever_config["collection_name"],
        )

        retriever = vector_store.as_retriever(
            search_type=retriever_config["search_type"],
            search_kwargs=retriever_config["search_kwargs"]
        )
        return retriever

    return _get_retriever_chain
