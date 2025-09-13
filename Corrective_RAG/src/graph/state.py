from typing import Annotated, List
from typing_extensions import TypedDict
from langchain_core.documents import Document


# Define State
class GraphState(TypedDict):
    question: Annotated[str, "The question to answer"]
    generation: Annotated[str, "The generation from the LLM"]
    web_search: Annotated[str, "Whether to add search"]
    documents: Annotated[List[Document], "The documents retrieved"]