from langgraph.graph import StateGraph, END
from langgraph.graph.state import CompiledStateGraph

from .state import State
from .nodes import get_retriever_node, get_answer_generation_node


def create_graph(config: dict) -> CompiledStateGraph:
    """
    Create and compile a workflow graph for a question-answering system.

    Args:
        config (dict): A dictionary containing configuration details for the workflow nodes.
            - "retriever": Configuration for the retriever node.
            - "generative_model": Configuration for the generative model node.

    Returns:
        A compiled workflow application instance.
    """
    # Initialize the state graph
    workflow = StateGraph(State)

    # Create the retriever node
    retriever_node = get_retriever_node(retriever_config=config["retriever"])
    workflow.add_node("retriever", retriever_node)
    workflow.set_entry_point("retriever")

    # Create the answer generation node
    answer_generation_node = get_answer_generation_node(llm_config=config["generative_model"])
    workflow.add_node("answer_generation", answer_generation_node)

    # Define edges between nodes
    workflow.add_edge("retriever", "answer_generation")
    workflow.add_edge("answer_generation", END)

    # Compile the graph into an application
    app = workflow.compile(debug=False)

    # Uncomment the following block to generate a visual representation of the graph
    # plot = app.get_graph().draw_mermaid_png()
    # with open("plot.png", "wb") as fp:
    #     fp.write(plot)

    return app
