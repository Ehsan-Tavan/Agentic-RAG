from typing import Dict
from langchain_text_splitters import TextSplitter, RecursiveCharacterTextSplitter


def get_chunker(chunker_config: Dict[str, int]) -> TextSplitter:
    """
    Initialize and return a RecursiveCharacterTextSplitter based on the provided configuration.

    Args:
        chunker_config : A dictionary containing the configuration for the chunker.
            - "chunk_size": The size of each text chunk.
            - "chunk_overlap": The number of overlapping characters between consecutive chunks.

    Returns:
        A text splitter configured with the provided chunk size and overlap.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunker_config["chunk_size"],
        chunk_overlap=chunker_config["chunk_overlap"],
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter
