from typing import Dict, Union
from langgraph.graph import StateGraph, START, END
from Research_AI_Agent.src.graph.state import SectionState, SectionOutputState
from .nodes.generate_section_research_queries import get_generate_section_research_queries_node
from .nodes.web_search import get_web_search_node
from .nodes.section_writer import get_section_writer_node


def create_section_builder_sub_agent(config: Dict[str, Union[str, int, float, Dict[str, str]]],
                                     tavily_search):
    generate_queries_node = get_generate_section_research_queries_node(
        model_config=config["model_config"],
        number_of_queries=config["web_search_config"]["number_of_queries_for_section"])

    search_web_node = get_web_search_node(web_search_config=config["web_search_config"],
                                          tavily_search=tavily_search)
    write_section_node = get_section_writer_node(model_config=config["model_config"])

    section_builder = StateGraph(SectionState, output=SectionOutputState)

    section_builder.add_node("generate_queries", generate_queries_node)
    section_builder.add_node("search_web", search_web_node)
    section_builder.add_node("write_section", write_section_node)

    section_builder.add_edge(START, "generate_queries")
    section_builder.add_edge("generate_queries", "search_web")
    section_builder.add_edge("search_web", "write_section")
    section_builder.add_edge("write_section", END)
    section_builder_agent = section_builder.compile()

    plot = section_builder_agent.get_graph().draw_mermaid_png()
    with open("../images/section_builder_agent.png", "wb") as fp:
        fp.write(plot)

    return section_builder_agent
