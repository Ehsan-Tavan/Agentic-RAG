from ..state import State


class RetrieverNode:
    def __init__(self, retriever_chain):
        self.retriever_chain = retriever_chain

    def __call__(self, state: State):
        return {
            "context": self.retriever_chain.invoke(state["question"])
        }


