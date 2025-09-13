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
        db_name: str,
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
        db_name: Name of the Milvus database to store data.
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

        doc_splits = split_documents(documents=documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

        # Generate embeddings
        texts = [doc.page_content for doc in doc_splits]
        metadatas = [doc.metadata for doc in doc_splits]
        embeddings = embedding_model.embed_documents(texts)

        # Connect to Milvus
        client = MilvusClient(uri=f"http://{host}:{port}",
                              db_name=db_name)

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

def split_documents(
        documents: List[Document],
        chunk_size: int = 100,
        chunk_overlap: int = 50
) -> List[Document]:
    """
    Splits a list of document lists into smaller chunks using recursive character splitting.

    Args:
        documents: A list of lists containing Document objects to be split.
        chunk_size: The maximum size of each chunk (in characters). Defaults to 100.
        chunk_overlap: The number of characters to overlap between chunks. Defaults to 50.

    Returns:
        A list of Document objects, each representing a chunk of the original documents.

    Raises:
        ValueError: If the input documents list is empty, invalid, or if chunk parameters are invalid.
        TypeError: If the input contains non-Document objects.
    """
    # Input validation
    if not documents:
        raise ValueError("Input documents list cannot be empty")

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap cannot be negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be less than chunk_size")

    # Flatten the nested list and validate document types
    try:
        flattened_docs = [
            doc for doc in documents
            if isinstance(doc, Document)
        ]
    except TypeError as e:
        raise TypeError("All items must be Document objects") from e

    if not flattened_docs:
        raise ValueError("No valid Document objects found in input")

    # Initialize text splitter with provided parameters
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        add_start_index=True,
        strip_whitespace=True
    )

    # Split documents and return
    return text_splitter.split_documents(flattened_docs)