from typing import Optional, Dict, Literal
from langchain.vectorstores import Milvus, VectorStoreRetriever
from langchain.embeddings.base import Embeddings
from .embedding import EmbeddingFactory


class MilvusVectorStore:
    @staticmethod
    def create(
            embedding: Embeddings,
            database_name: str,
            collection_name: str,
            host: str,
            port: str,
            text_field: str = "content",
    ) -> Milvus:
        return Milvus(
            embedding_function=embedding,
            collection_name=collection_name,
            connection_args={
                "db_name": database_name,
                "host": host,
                "port": port
            },
            text_field=text_field,
        )


class MilvusRetriever:
    @staticmethod
    def create(
            vectorstore: Milvus,
            search_kwargs: Optional[Dict] = None
    ) -> VectorStoreRetriever:
        return vectorstore.as_retriever(search_kwargs=search_kwargs or {"k": 4})


def get_milvus_retriever(
        database_name: str = "default",
        collection_name: str = "langchain_collection",
        host: str = "localhost",
        port: str = "19530",
        model_type: Literal["openai", "sentence-transformer"] = "openai",
        embedding_model: str = None,
        search_kwargs: Optional[Dict] = None,
) -> VectorStoreRetriever:
    """
    Create a retriever from an existing Milvus vector store.
    """

    embedding = EmbeddingFactory.create(model_type=model_type, model_name=embedding_model)
    vectorstore = MilvusVectorStore.create(
        embedding, database_name, collection_name, host, port
    )
    return MilvusRetriever.create(vectorstore, search_kwargs)
