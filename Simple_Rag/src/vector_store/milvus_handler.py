from typing import List
from pymilvus import MilvusClient, FieldSchema, CollectionSchema, DataType
from Simple_Rag.src.embeddings import HuggingFaceEmbeddingModel
from langchain.docstore.document import Document


class MilvusHandler:
    """
    A handler class for managing a Milvus vector database for text embeddings.
    """

    def __init__(
            self,
            vector_db_config: dict,
            retriever_model_config: dict
    ) -> None:
        """
        Initialize the MilvusHandler instance.

        Args:
            vector_db_config: Configuration for the Milvus vector database,
                including connection URI and collection name.
            retriever_model_config: Configuration for the embedding model,
                including model name and keyword arguments.
        """
        try:
            # Initialize the Milvus client
            self.client = MilvusClient(
                uri=vector_db_config["connection_args"]["uri"],
                db_name=vector_db_config["db_name"],
            )
        except Exception as e:
            raise RuntimeError(f"Failed to initialize Milvus client: {e}")

        # Collection-specific settings
        self.collection_name = vector_db_config["collection_name"]

        # Initialize the embedding model
        self.embedding_model = HuggingFaceEmbeddingModel(
            model_name=retriever_model_config["model_name"],
            model_kwargs=retriever_model_config["model_kwargs"]
        )

    def create_collection(
            self
    ) -> None:
        """
        Create a collection in the Milvus database for storing text embeddings.

        The collection schema includes:
            - `id`: Auto-generated primary key (INT64).
            - `vector`: Embedding vector (FLOAT_VECTOR).
            - `text`: Original text (VARCHAR).

        If the collection already exists, this method does nothing.
        """
        if self.client.has_collection(collection_name=self.collection_name):
            return

        try:
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
        except Exception as e:
            raise RuntimeError(f"Failed to create collection '{self.collection_name}': {e}")

    def insert_documents(
            self,
            documents: List[Document],
            batch_size: int = 100
    ) -> None:
        """
        Insert documents into the Milvus collection after embedding.

        Args:
            documents: A list of LangChain `Document` objects.
            batch_size: The number of documents to insert in each batch. Defaults to 100.

        Raises:
            RuntimeError: If the collection does not exist.
        """
        if not self.client.has_collection(collection_name=self.collection_name):
            raise RuntimeError(
                f"Collection '{self.collection_name}' does not exist. "
                f"Please create the collection first using `create_collection()`."
            )

        try:
            # Embed documents
            vectors = self.embedding_model.embed_documents([doc.page_content for doc in documents])
            data = [
                {
                    "vector": vectors[i],
                    "text": documents[i].page_content,
                }
                for i in range(len(documents))
            ]

            # Batch insertion
            for i in range(0, len(data), batch_size):
                batch_data = data[i:i + batch_size]
                self.client.insert(collection_name=self.collection_name, data=batch_data)
        except Exception as e:
            raise RuntimeError(f"Failed to insert documents into collection '{self.collection_name}': {e}")
