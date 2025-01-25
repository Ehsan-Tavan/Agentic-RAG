import os
from typing import List, Callable
from langchain.docstore.document import Document
from markitdown import MarkItDown


def is_valid_file(file_path: str) -> bool:
    valid_extensions = {".docx", ".pdf"}  # Using a set for faster lookups
    return os.path.splitext(file_path)[1].lower() in valid_extensions


def get_data_loader() -> Callable[[str], List[Document]]:
    md = MarkItDown()

    def _loader(file_path: str) -> List[Document]:
        if not is_valid_file(file_path):
            raise ValueError("Invalid file type. Only DOCX and PDF files are allowed.")
        loaded_file = md.convert(file_path)
        return [Document(page_content=loaded_file.text_content)]

    return _loader
