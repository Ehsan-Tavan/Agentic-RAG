import os
from typing import List
import json
from langchain_core.documents import Document

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