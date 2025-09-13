import os
from typing import List
import json
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import SentenceTransformerEmbeddings
from typing import Union, Optional
import logging


def load_documents_from_json(input_file_path: str) -> List[Document]:
    """
    Load content from a JSON file and convert it to LangChain Document objects.

    Args:
        input_file_path: Path to the JSON file to load.

    Returns:
        List of LangChain Document objects.

    Raises:
        FileNotFoundError: If the input file does not exist.
        json.JSONDecodeError: If the JSON file is invalid.
    """
    if not os.path.exists(input_file_path):
        raise FileNotFoundError(f"The file {input_file_path} does not exist.")

    with open(input_file_path, "r", encoding="utf-8") as f:
        loaded_data = json.load(f)

    # Convert to Document objects
    return [Document(page_content=item["page_content"], metadata=item["metadata"]) for item in loaded_data]


def load_embedding_model(
        model_name: str = "all-MiniLM-L6-v2") -> SentenceTransformerEmbeddings:
    """
    Loads an embedding model compatible with LangChain (SentenceTransformer).


    Args:
        model_name (str): Name of the model to load (default: 'all-MiniLM-L6-v2').

    Returns:
        SentenceTransformerEmbeddings: Loaded embedding model.

    Raises:
        ValueError: If the model fails to load.
    """
    try:
        model = SentenceTransformerEmbeddings(model_name=model_name)
        print(f"Loaded SentenceTransformerEmbeddings: {model_name}")
        return model
    except Exception as e:
        raise ValueError(f"Failed to load model {model_name}: {str(e)}")


def get_embedding_size(embedding_model: SentenceTransformerEmbeddings) -> int:
    """
    Gets the embedding size (dimensionality) of the model.

    Args:
        embedding_model (SentenceTransformerEmbeddings): Loaded embedding model.

    Returns:
        int: The size of the embedding vector.

    Raises:
        ValueError: If the model is invalid or embedding size cannot be determined.
    """
    try:
        # Encode a dummy sentence to get the embedding size
        sample_text = "sample text"
        embedding = embedding_model.embed_query(sample_text)
        dimension = len(embedding)
        print(f"Detected SentenceTransformerEmbeddings, dimension: {dimension}")
        return dimension
    except Exception as e:
        raise ValueError(f"Failed to determine embedding size: {str(e)}")
