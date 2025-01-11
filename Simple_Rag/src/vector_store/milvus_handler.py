from typing import List
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from Simple_Rag.src.embeddings import HuggingFaceEmbeddingModel
from langchain.docstore.document import Document


class MilvusHandler:
    def __init__(
            self,
            vector_db_config: dict,
            retriever_model_config: dict
    ):
        # Initialize the Milvus client
        self.client = MilvusClient(
            uri=vector_db_config["connection_args"]["uri"],
            db_name=vector_db_config["db_name"],
        )
        # Collection-specific settings
        self.collection_name = vector_db_config["collection_name"]
        # Initialize the embedding model
        self.embedding_model = HuggingFaceEmbeddingModel(
            model_name=retriever_model_config["model_name"],
            model_kwargs=retriever_model_config["model_kwargs"]
        )

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            return

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR,
                            dim=self.embedding_model.get_dimension()),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024),
            ],
            description="Collection for text embeddings with vectors and metadata",
        )
        self.client.create_collection(collection_name=self.collection_name, schema=schema)

    def insert_documents(
            self,
            documents: List[Document]
    ) -> None:
        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError(
                f"Collection '{self.collection_name}' does not exist. "
                f"Please create the collection first using `create_collection()`."
            )

        vectors = self.embedding_model.embed_documents([data.page_content for data in documents])
        data = [
            {
                "vector": vectors[i],
                "text": documents[i].page_content,
            }
            for i in range(len(documents))
        ]
        self.client.insert(collection_name=self.collection_name, data=data)
