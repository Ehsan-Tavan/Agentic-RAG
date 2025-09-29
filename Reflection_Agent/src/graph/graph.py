from langgraph.graph import StateGraph, START, END
from langchain_core.runnables.graph_mermaid import MermaidDrawMethod
import pyppeteer

from Reflection_Agent.src.graph.state import ReflectionState
from Reflection_Agent.src.graph.nodes import get_generation_node, get_critic_node, get_refiner_node


def create_graph(config: dict):
    workflow = StateGraph(ReflectionState)
    generation_node = get_generation_node(config["llm"])
    critic_node = get_critic_node(config["llm"])
    refiner_node = get_refiner_node(config["llm"])

    workflow.add_node("generation", generation_node)
    workflow.add_node("critic", critic_node)
    workflow.add_node("refiner", refiner_node)

    workflow.add_edge(START, "generation")
    workflow.add_edge("generation", "critic")
    workflow.add_edge("critic", "refiner")
    workflow.add_edge("refiner", END)

    app = workflow.compile()

    # plot = app.get_graph(xray=True).draw_mermaid_png(
    #     draw_method=MermaidDrawMethod.PYPPETEER,
    # )
    # with open("../images/reflection_agent.png", "wb") as fp:
    #     fp.write(plot)

    return app
