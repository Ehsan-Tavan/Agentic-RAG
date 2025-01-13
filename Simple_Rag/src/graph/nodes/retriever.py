from Simple_Rag.src.graph.state import State
from Simple_Rag.src.retriever import create_retriever_chain


class RetrieverNode:
    def __init__(self, retriever_chain):
        self.retriever_chain = retriever_chain

    def __call__(self, state: State):
        return {
            "context": self.retriever_chain.invoke(state["question"])
        }


def get_retriever_node(retriever_config):
    retriever_chain = create_retriever_chain(retriever_config)
    return RetrieverNode(retriever_chain())
