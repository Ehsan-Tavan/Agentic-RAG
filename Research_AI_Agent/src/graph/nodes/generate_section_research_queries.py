from typing import Dict

from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.structures import Queries
from Research_AI_Agent.src.graph.state import SectionState
from Research_AI_Agent.src.graph.promps import generate_section_research_queries_prompt_creator


class GenerateSectionResearchQueriesNode:
    def __init__(self, chain):
        self.chain = chain

    def __call__(self, state: SectionState) -> Dict[str, Queries]:
        print(f"--- Generating Search Queries for Section: {state['section'].name} ---")

        # Retrieve section details
        section = state["section"]

        search_queries = self.chain.invoke({"section_topic": section.description})

        return {"research_queries": search_queries.queries}


def get_generate_section_research_queries_chain(model_config: Dict[str, str], number_of_queries: int = 5):
    prompt_template = generate_section_research_queries_prompt_creator(number_of_queries=number_of_queries)
    llm = load_model(model_config)
    structured_llm = llm.with_structured_output(Queries)
    chain = prompt_template | structured_llm
    return chain


def get_generate_section_research_queries_node(model_config: Dict[str, str], number_of_queries: int = 5):
    generate_section_research_queries_chain = get_generate_section_research_queries_chain(
        model_config=model_config, number_of_queries=number_of_queries)
    generate_section_research_queries_node = GenerateSectionResearchQueriesNode(
        chain=generate_section_research_queries_chain
    )
    return generate_section_research_queries_node
