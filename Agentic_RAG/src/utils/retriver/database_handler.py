from typing import List, Optional, Union
from pymilvus import connections, MilvusClient, CollectionSchema, FieldSchema, DataType
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def create_milvus_database_and_collection(
        database_name: str,
        collection_name: str,
        host: str = "localhost",
        port: str = "19530",
        dimension: int = 384  # Dimension for multilingual-e5-small
) -> MilvusClient:
    """
    Create a Milvus database and a collection with a predefined schema.

    Args:
        database_name: Name of the database to create.
        collection_name: Name of the collection to create.
        host: Milvus server host address.
        port: Milvus server port.
        dimension: Dimension of the vector embeddings (384 for multilingual-e5-small).

    Returns:
        MilvusClient instance connected to the specified database.

    Raises:
        Exception: If database or collection creation fails.
    """
    try:
        # Connect to Milvus
        connections.connect(alias="default", host=host, port=port)
        client = MilvusClient(uri=f"http://{host}:{port}")

        # Create database if it doesn't exist
        if database_name not in client.list_databases():
            client.create_database(database_name)
            print(f"Created database: {database_name}")
        else:
            print(f"Database {database_name} already exists")

        # Switch to the database
        client.using_database(database_name)

        # Define collection schema
        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=65535),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension)
            ],
            description=f"Collection for {database_name}",
            enable_dynamic_field=True
        )

        # Create collection if it doesn't exist
        if collection_name not in client.list_collections():
            client.create_collection(
                collection_name=collection_name,
                schema=schema,
                shards_num=2
            )
            print(f"Created collection: {collection_name}")
        else:
            print(f"Collection {collection_name} already exists")

        return client

    except Exception as e:
        print(f"Failed to create database/collection: {str(e)}")
        raise


def save_data_to_milvus(
        documents: List[Document],
        collection_name: str,
        host: str = "localhost",
        port: str = "19530",
        embedding_model: Optional[Union[OpenAIEmbeddings, SentenceTransformerEmbeddings]] = None,
        chunk_size: int = 100,
        chunk_overlap: int = 50
) -> None:
    """
    Split documents, generate embeddings, and save to a Milvus collection.

    Args:
        documents: List of LangChain Document objects to process.
        collection_name: Name of the Milvus collection to store data.
        host: Milvus server host address.
        port: Milvus server port.
        embedding_model: Optional embedding model. If None, uses OpenAIEmbeddings.
        chunk_size: Size of document chunks for splitting.
        chunk_overlap: Overlap between document chunks.
    """
    try:
        # Initialize embeddings if not provided
        if embedding_model is None:
            embedding_model = OpenAIEmbeddings()

        # Split documents
        text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
            chunk_size=chunk_size, chunk_overlap=chunk_overlap
        )
        doc_splits = text_splitter.split_documents(documents)

        # Generate embeddings
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [doc.metadata for doc in doc_splits]
        embeddings = embedding_model.embed_documents(texts)

        # Connect to Milvus
        client = MilvusClient(uri=f"http://{host}:{port}")

        # Insert data into collection
        data = [
            {
                "content": text,
                "metadata": metadata,
                "vector": embedding
            }
            for text, metadata, embedding in zip(texts, metadatas, embeddings)
        ]

        client.insert(collection_name=collection_name, data=data)
        print(f"Inserted {len(data)} documents into collection {collection_name}")

    except Exception as e:
        print(f"Failed to save data to Milvus: {str(e)}")
        raise
