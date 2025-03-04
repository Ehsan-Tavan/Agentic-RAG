from typing import Dict, List, Union
import logging
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from Research_AI_Agent.src.graph.state import ReportState, SectionState
from Research_AI_Agent.src.graph.helper import run_search_queries, format_search_query_results

logger = logging.getLogger(__name__)


class WebSearchNode:
    """
    A class responsible for performing web searches based on research queries.
    """

    def __init__(
            self,
            num_results: int,
            max_tokens: int,
            include_raw_content: bool,
            tavily_search: TavilySearchAPIWrapper
    ):
        """
        Initializes the WebSearchNode with search configuration and Tavily client.

        Args:
            num_results: Number of search results to retrieve.
            max_tokens: Maximum number of tokens for formatting search results.
            include_raw_content: Whether to include raw content in the search results.
            tavily_search: The Tavily search client.
        """
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.include_raw_content = include_raw_content
        self.tavily_search = tavily_search

    async def perform_search(
            self,
            query_list: List[str]
    ) -> str:
        """
        Performs a web search using the provided list of queries.

        Args:
            query_list: List of research queries to search for.

        Returns:
            Formatted search results or a default message if no results are found.
        """
        logger.info("Performing web search.")
        search_docs = await run_search_queries(
            tavily_search=self.tavily_search,
            search_queries=query_list,
            num_results=self.num_results,
            include_raw_content=self.include_raw_content
        )

        if not search_docs:
            logger.warning("No search results returned.")
            return "No search results available."

        return format_search_query_results(
            search_docs,
            max_tokens=self.max_tokens,
            include_raw_content=self.include_raw_content
        )

    async def __call__(
            self,
            state: Union[ReportState, SectionState]
    ) -> Dict[str, str]:
        """
        Executes the web search node and returns the search context.

        Args:
            state: The state containing research queries.

        Returns:
            A dictionary containing the formatted search context.
        """
        logger.info(" --- Web Search Node --- ")
        query_list = state["research_queries"]
        search_context = await self.perform_search(query_list)
        logger.debug({
            "message": "Search context generated",
            "search_context": search_context})
        return {"search_context": search_context}


def get_web_search_node(
        web_search_config: Dict[str, Union[str, int, bool]],
        tavily_search: TavilySearchAPIWrapper
) -> WebSearchNode:
    """
    Creates and returns an instance of the `WebSearchNode` class.

    Args:
        web_search_config: A dictionary containing the
            configuration for the web search. Expected keys are:
            - "num_results" (int): The number of search results to retrieve.
            - "max_tokens" (int): The maximum number of tokens for formatting search results.
            - "include_raw_content" (bool): Whether to include raw content in the search results.
        tavily_search: The Tavily search client used to perform the web searches.

    Returns:
        An instance of the `WebSearchNode` class configured with the provided search parameters and Tavily client.
    """
    logger.info("Creating the `web search node`.")
    web_search_node = WebSearchNode(num_results=web_search_config["num_results"],
                                    max_tokens=web_search_config["max_tokens"],
                                    include_raw_content=web_search_config["include_raw_content"],
                                    tavily_search=tavily_search)
    logger.info("The `web search node` has been created.")
    return web_search_node
