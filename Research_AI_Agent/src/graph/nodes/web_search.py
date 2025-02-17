from typing import Dict
from Research_AI_Agent.src.graph.state import ReportState
from Research_AI_Agent.src.graph.helper import run_search_queries, format_search_query_results


class WebSearchNode:
    def __init__(self, num_results, max_tokens, include_raw_content):
        self.num_results = num_results
        self.max_tokens = max_tokens
        self.include_raw_content = include_raw_content

    async def __call__(self, state: ReportState) -> Dict[str, str]:
        print(" --- Performing Web Search --- ")
        query_list = state["research_queries"]
        search_docs = await run_search_queries(
            query_list,
            num_results=self.num_results,
            include_raw_content=self.include_raw_content
        )
        if not search_docs:
            print("Warning: No search results returned")
            search_context = "No search results available."
        else:
            search_context = format_search_query_results(
                search_docs,
                max_tokens=self.max_tokens,
                include_raw_content=self.include_raw_content
            )

        return {
            "search_context": search_context
        }


def get_web_search_node(web_search_config: Dict[str, str]):
    return WebSearchNode(num_results=web_search_config["num_results"],
                         include_raw_content=web_search_config["include_raw_content"])
