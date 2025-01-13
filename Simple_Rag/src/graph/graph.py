from langgraph.graph import StateGraph, END

from .state import State
from .nodes import get_retriever_node, get_answer_generation_node


def create_graph(config):
    workflow = StateGraph(State)
    retriever_node = get_retriever_node(retriever_config=config["retriever"])
    answer_generation_node = get_answer_generation_node(llm_config=config["generative_model"])

    workflow.add_node("retriever", retriever_node)
    workflow.set_entry_point("retriever")

    workflow.add_node("answer_generation", answer_generation_node)

    workflow.add_edge("retriever", "answer_generation")

    workflow.add_edge("answer_generation", END)

    app = workflow.compile(debug=False)

    # plot = app.get_graph().draw_mermaid_png()
    #
    # with open("plot.png", "wb") as fp:
    #     fp.write(plot)

    return app


