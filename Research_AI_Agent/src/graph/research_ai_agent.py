from typing import Dict, Union
import logging
from langgraph.graph import StateGraph, START, END
from langgraph.constants import Send

from .state import ReportState, ReportStateInput, ReportStateOutput
from .nodes.generate_topic_research_queries import get_generate_topic_research_queries_node
from .nodes.report_plan_section_generator import get_generate_report_plan_sections_node
from .section_builder_agent import create_section_builder_sub_agent
from .nodes.section_formatter import get_section_formatter_node
from .nodes.finalize_sections_content import get_final_section_writer_node
from .nodes.final_report_compiler import get_final_report_compiler_node
from .nodes.web_search import get_web_search_node

logger = logging.getLogger(__name__)


def parallelize_section_writing(state: ReportState):
    # Kick off section writing in parallel via Send() API for any sections that require research
    return [
        Send("section_builder_with_web_search",  # name of the subagent node
             {"section": s})
        for s in state["sections"]
        if s.research
    ]


def parallelize_final_section_writing(state: ReportState):
    # Kick off section writing in parallel via Send() API for any sections that do not require research
    return [
        Send("write_final_sections",
             {"section": s, "report_sections_from_research": state["report_sections_from_research"]})
        for s in state["sections"]
        if not s.research
    ]


def create_reporter_agent(config: Dict[str, Union[str, int, float, Dict[str, str]]],
                          tavily_search):
    logger.info("Start creating reporter agent.")
    generate_topic_research_queries_node = get_generate_topic_research_queries_node(
        model_config=config["model_config"],
        number_of_queries=config["number_of_queries"])
    web_search_node = get_web_search_node(web_search_config=config["web_search_config"],
                                          tavily_search=tavily_search)
    generate_report_plan_node = get_generate_report_plan_sections_node(model_config=config["model_config"])
    section_builder_agent = create_section_builder_sub_agent(config=config, tavily_search=tavily_search)
    section_formatter_node = get_section_formatter_node()
    final_section_writer_node = get_final_section_writer_node(model_config=config["model_config"])
    final_report_compiler_node = get_final_report_compiler_node()

    builder = StateGraph(state_schema=ReportState, input=ReportStateInput, output=ReportStateOutput)
    builder.add_node("generate_topic_research_queries", generate_topic_research_queries_node)
    builder.add_node("web_search", web_search_node)
    builder.add_node("generate_report_plan", generate_report_plan_node)
    builder.add_node("section_builder_with_web_search", section_builder_agent)
    builder.add_node("format_completed_sections", section_formatter_node)
    builder.add_node("write_final_sections", final_section_writer_node)
    builder.add_node("compile_final_report", final_report_compiler_node)

    builder.add_edge(START, "generate_topic_research_queries")
    builder.add_edge("generate_topic_research_queries", "web_search")
    builder.add_edge("web_search", "generate_report_plan")

    builder.add_conditional_edges("generate_report_plan",
                                  parallelize_section_writing,
                                  ["section_builder_with_web_search"])
    builder.add_edge("section_builder_with_web_search", "format_completed_sections")
    builder.add_conditional_edges("format_completed_sections",
                                  parallelize_final_section_writing,
                                  ["write_final_sections"])
    builder.add_edge("write_final_sections", "compile_final_report")
    builder.add_edge("compile_final_report", END)

    reporter_agent = builder.compile()

    plot = reporter_agent.get_graph(xray=True).draw_mermaid_png()
    with open("final_plot.png", "wb") as fp:
        fp.write(plot)

    return reporter_agent
