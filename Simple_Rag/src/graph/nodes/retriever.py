from typing import Dict
from Simple_Rag.src.graph.state import State
from Simple_Rag.src.retriever import create_retriever_chain


class RetrieverNode:
    """
    Node responsible for retrieving relevant context for a given question.

    Args:
        retriever_chain: A callable chain that retrieves relevant context based on a question.
    """

    def __init__(self, retriever_chain):
        self.retriever_chain = retriever_chain

    def __call__(
            self,
            state: State
    ) -> Dict[str, list]:
        """
         Retrieve context based on the question in the state.

         Args:
             state: The current state containing the question.

         Returns:
             A dictionary with the key "context" containing the retrieved context.
         """
        return {
            "context": self.retriever_chain.invoke(state["question"])
        }


def get_retriever_node(retriever_config: dict) -> RetrieverNode:
    """
    Create an instance of RetrieverNode using the specified retriever configuration.

    Args:
        retriever_config: Configuration details for initializing the retriever chain.

    Returns:
        An instance of RetrieverNode configured with the retriever chain.
    """
    retriever_chain = create_retriever_chain(retriever_config)
    return RetrieverNode(retriever_chain())
