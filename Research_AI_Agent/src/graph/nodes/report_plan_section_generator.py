from typing import Dict
from Research_AI_Agent.src.graph.promps import generate_report_plan_sections_prompt_creator, REPORT_STRUCTURE
from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.structures import Sections
from Research_AI_Agent.src.graph.state import ReportState


class GenerateReportPlanSectionNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state: ReportState) -> Dict[str, Sections]:
        print(" --- Generating Report Plan Sections --- ")
        topic = state["topic"]
        search_context = state["search_context"]

        report_sections = self.chain.invoke({"topic": topic, "search_context": search_context})

        return {
            "sections": report_sections
        }


def get_generate_report_plan_sections_chain(model_config: Dict[str, str]):
    prompt_template = generate_report_plan_sections_prompt_creator(report_organization=REPORT_STRUCTURE)
    llm = load_model(model_config)
    structured_llm = llm.with_structured_output(Sections)
    chain = prompt_template | structured_llm
    return chain


def get_generate_report_plan_sections_node(model_config: Dict[str, str]):
    generate_report_plan_chain = get_generate_report_plan_sections_chain(model_config)
    generate_report_plan_node = GenerateReportPlanSectionNode(
        chain=generate_report_plan_chain
    )
    return generate_report_plan_node
