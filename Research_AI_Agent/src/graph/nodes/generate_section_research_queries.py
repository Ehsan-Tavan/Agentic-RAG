from typing import Dict, List
import logging

from Research_AI_Agent.src.graph.helper import load_model
from Research_AI_Agent.src.graph.structures import Queries, SearchQuery
from Research_AI_Agent.src.graph.state import SectionState
from Research_AI_Agent.src.graph.promps import generate_section_research_queries_prompt_creator

logger = logging.getLogger(__name__)


class SectionResearchQueryGenerator:
    """
    A class responsible for generating research queries for a specific section.
    """

    def __init__(
            self,
            model_config: Dict[str, str],
            number_of_queries: int = 5
    ):
        """
        Initializes the SectionResearchQueryGenerator with the given model configuration.

        Args:
            model_config: Configuration for the language model.
            number_of_queries: Number of research queries to generate. Defaults to 5.
        """
        self.model_config = model_config
        self.number_of_queries = number_of_queries
        self.chain = self._initialize_chain()

    def _initialize_chain(
            self
    ):
        """
        Initializes the chain for generating section research queries.

        Returns:
            The configured chain for query generation.
        """
        logger.info("Initializing the section research query generation chain.")
        prompt_template = generate_section_research_queries_prompt_creator(
            number_of_queries=self.number_of_queries
        )
        llm = load_model(self.model_config)
        structured_llm = llm.with_structured_output(Queries)
        return prompt_template | structured_llm

    def generate_queries(
            self, section_topic: str
    ) -> List[str]:
        """
        Generates a list of research queries for the given section topic.

        Args:
            section_topic: The topic of the section for which queries are generated.

        Returns:
            A list of generated research queries.
        """
        logger.info("Generating section research queries.")
        research_queries = self.chain.invoke({"section_topic": section_topic})
        query_list = [
            query.search_query if isinstance(query, SearchQuery) else str(query)
            for query in research_queries.queries
        ]
        return query_list


class GenerateSectionResearchQueriesNode:
    """
    A class responsible for generating research queries for a section as part of a node in a graph-based workflow.
    """

    def __init__(
            self,
            section_research_query_generator: SectionResearchQueryGenerator
    ):
        """
        Initializes the GenerateSectionResearchQueriesNode with a SectionResearchQueryGenerator.

        Args:
            section_research_query_generator: An instance of SectionResearchQueryGenerator.
        """
        self.section_research_query_generator = section_research_query_generator

    def __call__(
            self,
            state: SectionState
    ) -> Dict[str, List[str]]:
        """
        Generates research queries for the section in the given state.

        Args:
            state: The state containing the section for which queries are generated.

        Returns:
            A dictionary containing the generated research queries.
        """
        logger.info(f"--- Generating Search Queries for Section: {state['section'].name} ---")
        section = state["section"]
        result = self.section_research_query_generator.generate_queries(section.description)
        logger.debug({"message": "section_research_queries have been generated",
                      "section_name": section.name,
                      "research_queries": result})
        return {"research_queries": result}


def get_generate_section_research_queries_node(
        model_config: Dict[str, str],
        number_of_queries: int = 5
):
    """
    Creates and returns an instance of GenerateSectionResearchQueriesNode.

    Args:
        model_config: Configuration for the language model.
        number_of_queries: Number of research queries to generate. Defaults to 5.

    Returns:
        An instance of GenerateSectionResearchQueriesNode.
    """
    logger.info("Creating the `generate section research queries node`.")
    section_research_query_generator = SectionResearchQueryGenerator(model_config, number_of_queries)
    generate_section_research_queries_node = GenerateSectionResearchQueriesNode(
        section_research_query_generator=section_research_query_generator
    )
    logger.info("The `generate section research queries node` has been created.")
    return generate_section_research_queries_node
