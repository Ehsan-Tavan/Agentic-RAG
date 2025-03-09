from typing import Dict, List
import logging

from ..state import ReportState
from ..promps import generate_topic_research_queries_prompt_creator, REPORT_STRUCTURE
from ..helper import load_model
from ..structures import Queries, SearchQuery

logger = logging.getLogger(__name__)


class ResearchQueryGenerator:
    """
    A class responsible for generating research queries based on a given topic.
    """
    def __init__(
            self,
            model_config: Dict[str, str],
            number_of_queries: int
    ):
        """
        Initializes the ResearchQueryGenerator with the given model configuration and number of queries.

        Args:
            model_config: Configuration for the language model.
            number_of_queries: Number of research queries to generate. Defaults to 8.
        """
        self.model_config = model_config
        self.number_of_queries = number_of_queries
        self.chain = self._initialize_chain()

    def _initialize_chain(
            self
    ):
        """
        Initializes the chain for generating research queries.

        Returns:
            The configured chain for query generation.
        """
        logger.info("Initializing the research query generation chain.")
        prompt_template = generate_topic_research_queries_prompt_creator(
            number_of_queries=self.number_of_queries,
            report_organization=REPORT_STRUCTURE
        )
        llm = load_model(self.model_config)
        structured_llm = llm.with_structured_output(Queries)
        return prompt_template | structured_llm

    def generate_queries(
            self, topic: str
    ) -> List[str]:
        """
        Generates a list of research queries for the given topic.

        Args:
            topic: The topic for which research queries are generated.

        Returns:
           A list of generated research queries.
        """
        logger.info("Generating topic research queries.")
        research_queries = self.chain.invoke({"topic": topic})
        query_list = [
            query.search_query if isinstance(query, SearchQuery) else str(query)
            for query in research_queries.queries
        ]
        return query_list


class GenerateTopicResearchQueriesNode:
    """
    A class responsible for generating research queries as part of a node in a graph-based workflow.
    """
    def __init__(
            self,
            research_query_generator: ResearchQueryGenerator
    ):
        """
        Initializes the GenerateTopicResearchQueriesNode with a ResearchQueryGenerator.

        Args:
            research_query_generator: An instance of ResearchQueryGenerator.
        """
        self.research_query_generator = research_query_generator

    def __call__(
            self,
            state: ReportState
    ) -> Dict[str, List[str]]:
        """
        Generates research queries for the topic in the given state.

        Args:
            state: The state containing the topic for which queries are generated.

        Returns:
            A dictionary containing the generated research queries.
        """
        logger.info(" --- Generate Topic Research Queries Node --- ")
        topic = state["topic"]
        result = self.research_query_generator.generate_queries(topic)
        logger.debug({"message": "research_queries have been generated",
                      "topic": topic,
                      "research_queries": result})
        return {"research_queries": result}


def get_generate_topic_research_queries_node(
        model_config: Dict[str, str],
        number_of_queries: int
):
    """
    Creates and returns an instance of GenerateTopicResearchQueriesNode.

    Args:
        model_config: Configuration for the language model.
        number_of_queries: Number of research queries to generate. Defaults to 2.

    Returns:
        An instance of GenerateTopicResearchQueriesNode.
    """
    logger.info("Creating the `generate research queries node`.")
    research_query_generator = ResearchQueryGenerator(model_config, number_of_queries)
    generate_research_queries_node = GenerateTopicResearchQueriesNode(
        research_query_generator=research_query_generator
    )
    logger.info("The `generate research queries node` has been created.")
    return generate_research_queries_node
