from typing import Dict

from Research_AI_Agent.src.graph.state import ReportState
from Research_AI_Agent.src.graph.promps import generate_report_plan_prompt_creator, REPORT_ORGANIZATION
from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.structures import Queries


class GenerateResearchQueriesNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state: ReportState) -> Dict[str, Queries]:
        print(" --- Generating Research Queries --- ")
        topic = state["topic"]

        research_queries = self.chain.invoke({"topic": topic})

        return {
            "research_queries": research_queries
        }


def get_generate_research_queries_chain(model_config: Dict[str, str], number_of_queries: int = 8):
    prompt_template = generate_report_plan_prompt_creator(number_of_queries=number_of_queries,
                                                          report_organization=REPORT_ORGANIZATION)
    llm = load_model(model_config)
    structured_llm = llm.with_structured_output(Queries)
    chain = prompt_template | structured_llm
    return chain


def get_generate_research_queries_node(model_config: Dict[str, str]):
    generate_research_queries_chain = get_generate_research_queries_chain(model_config)
    generate_research_queries_node = GenerateResearchQueriesNode(
        chain=generate_research_queries_chain
    )
    return generate_research_queries_node
