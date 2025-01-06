from typing import TypedDict, List
from langchain_core.documents import Document


class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
