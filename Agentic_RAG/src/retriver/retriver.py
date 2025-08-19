from typing import List, Optional, Union
from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import VectorStoreRetriever

from langchain.tools.retriever import create_retriever_tool


def create_milvus_vectorstore(
        documents: List[Document],
        collection_name: str = "langchain_collection",
        host: str = "localhost",
        port: str = "19530",
        embedding_model: Optional[Union[OpenAIEmbeddings, SentenceTransformerEmbeddings]] = None,
        chunk_size: int = 100,
        chunk_overlap: int = 50
) -> Milvus:
    """
    Split documents, generate embeddings, and store in a Milvus vector store.

    Args:
        documents: List of LangChain Document objects to process.
        collection_name: Name of the Milvus collection to store vectors.
        host: Milvus server host address.
        port: Milvus server port.
        embedding_model: Optional embedding model. If None, uses OpenAIEmbeddings.
        chunk_size: Size of document chunks for splitting.
        chunk_overlap: Overlap between document chunks.

    Returns:
        Milvus vector store instance.
    """
    # Initialize embeddings if not provided
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()

    # Split documents
    text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    doc_splits = text_splitter.split_documents(documents)

    # Create Milvus vector store
    vectorstore = Milvus.from_documents(
        documents=doc_splits,
        embedding=embedding_model,
        collection_name=collection_name,
        connection_args={"host": host, "port": port}
    )

    return vectorstore


def get_milvus_retriever(
        database_name: str = "default",
        collection_name: str = "langchain_collection",
        host: str = "localhost",
        port: str = "19530",
        embedding_model: Optional[Union[OpenAIEmbeddings, SentenceTransformerEmbeddings]] = None,
        search_kwargs: Optional[dict] = None
) -> VectorStoreRetriever:
    """
    Create a retriever from an existing Milvus vector store.

    Args:
        database_name: Name of the Milvus database to use (default is "default").
        collection_name: Name of the Milvus collection to query.
        host: Milvus server host address.
        port: Milvus server port.
        embedding_model: Optional embedding model. If None, uses OpenAIEmbeddings.
        search_kwargs: Optional dictionary of search parameters (e.g., {"k": 4} for top-k results).

    Returns:
        VectorStoreRetriever instance for querying the Milvus vector store.

    Raises:
        ValueError: If the specified Milvus collection does not exist.
    """
    # Initialize embeddings if not provided
    if embedding_model is None:
        embedding_model = OpenAIEmbeddings()

    # Initialize Milvus vector store
    vectorstore = Milvus(
        embedding_function=embedding_model,
        collection_name=collection_name,
        connection_args={
            "db_name": database_name,
            "host": host,
            "port": port},
        text_field="content"

    )

    # Set default search kwargs if not provided
    if search_kwargs is None:
        search_kwargs = {"k": 4}  # Default to retrieving top 4 results

    # Create and return retriever
    return vectorstore.as_retriever(search_kwargs=search_kwargs)


def get_retriever_tool(retriever):
    retriever_tool = create_retriever_tool(
        retriever,
        "retrieve_blog_posts",
        "Search and return information about Lilian Weng blog posts.",
    )
    return retriever_tool
