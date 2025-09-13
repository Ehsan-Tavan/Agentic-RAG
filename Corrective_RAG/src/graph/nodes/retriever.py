from typing import Dict, List
from langchain.vectorstores.base import VectorStoreRetriever
from langchain_core.documents import Document

from Corrective_RAG.src.graph.state import GraphState
from Corrective_RAG.src.utils.retriever import get_milvus_retriever


class RetrieverNode:
    def __init__(self, retriever: VectorStoreRetriever):
        print("\n==== RETRIEVE ====\n")
        self.retriever = retriever

    def __call__(self, state: GraphState) -> Dict[str, List[Document]]:
        return {
            "documents": self.retriever.invoke(state["question"])
        }


def get_retriever_node(config: dict) -> RetrieverNode:
    milvus_retriever = get_milvus_retriever(database_name=config["database"]["db_name"],
                                            collection_name=config["database"]["collection_name"],
                                            host=config["database"]["host"],
                                            port=config["database"]["port"],
                                            model_type=config["retriever"]["embedding_type"],
                                            search_kwargs=config["retriever"]["search_kwargs"])
    return RetrieverNode(retriever=milvus_retriever)
