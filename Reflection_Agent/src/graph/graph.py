from langgraph.graph import StateGraph, START, END

from Reflection_Agent.src.graph.state import ReflectionState
from Reflection_Agent.src.graph.nodes import get_generation_node, get_critic_node


def create_graph(config: dict):
    workflow = StateGraph(ReflectionState)
    generation_node = get_generation_node(config["llm"])
    critic_node = get_critic_node(config["llm"])

    workflow.add_node("generation", generation_node)
    workflow.add_node("critic", critic_node)

    workflow.add_edge(START, "generation")
    workflow.add_edge("generation", "critic")
    workflow.add_edge("critic", END)

    app = workflow.compile()

    return app
