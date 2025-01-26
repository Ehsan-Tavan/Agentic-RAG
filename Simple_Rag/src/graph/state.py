from typing import TypedDict, List
from langchain_core.documents import Document


class State(TypedDict):
    """
    Represents the state in a question-answering workflow.

    Attributes:
        question: The question being processed.
        context: A list of documents providing context for answering the question.
        answer: The generated answer to the question.
    """
    question: str
    context: List[Document]
    answer: str
