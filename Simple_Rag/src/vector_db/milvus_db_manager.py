from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from langchain_huggingface import HuggingFaceEmbeddings


class EmbeddingModel:
    def __init__(self, retriever_model_config):
        self.model = HuggingFaceEmbeddings(
            model_name=retriever_model_config["model_name"],
            model_kwargs=retriever_model_config["model_kwargs"],
        )

    def get_dimension(self):
        return self.model._client.get_sentence_embedding_dimension()

    def embed_documents(self, documents):
        return self.model.embed_documents(documents)


class MilvusDatabase:
    def __init__(self, vector_db_config, dimension):
        print(vector_db_config["connection_args"]["uri"])
        self.client = MilvusClient(
            uri=vector_db_config["connection_args"]["uri"],
            db_name=vector_db_config["db_name"]
        )
        self.collection_name = vector_db_config["collection_name"]
        self.dimension = dimension

    def create_collection(self):
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=self.dimension),
                FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=1024)
            ],
            description="Collection for text embeddings with vectors and metadata"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema
        )

    def insert_data(self, data):
        self.client.insert(
            collection_name=self.collection_name,
            data=data
        )


class MilvusManager:
    def __init__(self, vector_db_config, retriever_model_config):
        self.embedding_model = EmbeddingModel(retriever_model_config=retriever_model_config)
        self.db = MilvusDatabase(
            vector_db_config=vector_db_config,
            dimension=self.embedding_model.get_dimension()
        )

    def insert_chunked_data(self, chunked_data):
        self.db.create_collection()
        vectors = self.embedding_model.embed_documents(
            [data.page_content for data in chunked_data]
        )
        data = [
            {
                "vector": vectors[i],
                "text": chunked_data[i].page_content,
            }
            for i in range(len(vectors))
        ]
        self.db.insert_data(data)
