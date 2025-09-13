from langgraph.graph import StateGraph, START, END


from Simple_Search_Agent.src.graph import State
from Simple_Search_Agent.src.graph.nodes import GenerationNode, SearchNode
from .tools import get_tools


def route_tools(
        state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END



def create_graph(configs):
    tools = get_tools(configs["search_tools"])
    graph_builder = StateGraph(State)
    generation_node = GenerationNode(llm_configs=configs["llm"], tools=tools)
    search_node = SearchNode(search_config=configs["search_tools"])

    graph_builder.add_node("chatbot", generation_node)
    graph_builder.add_node("tools", search_node)


    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        {"tools": "tools", END: END},
    )

    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")
    graph_builder.add_edge(START, "chatbot")
    graph = graph_builder.compile()

    # plot = graph.get_graph(xray=True).draw_mermaid_png()
    # with open("../images/search_agent.png", "wb") as fp:
    #     fp.write(plot)

    return graph
