import os
from typing import List, Union, Dict, Any
import asyncio
import tiktoken
from langchain_openai import ChatOpenAI
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_community.utilities.tavily_search import TavilySearchAPIWrapper

from Research_AI_Agent.src.graph.structures import SearchQuery

TAVILY_SEARCH = TavilySearchAPIWrapper()


def load_model(model_config) -> BaseChatOpenAI:
    llm = ChatOpenAI(
        model=model_config["name"],
        temperature=model_config["temperature"],
        api_key=os.getenv("OPENAI_API_KEY", model_config.get("api_key", None)),
        base_url=os.getenv("OPENAI_API_BASE", model_config.get("base_url", None))
    )

    return llm


async def run_search_queries(
        search_queries: List[Union[str, SearchQuery]],
        num_results: int = 5,
        include_raw_content: bool = False
) -> List[Dict]:
    search_tasks = []

    for query in search_queries:
        # Handle both string and SearchQuery objects
        # Just in case LLM fails to generate queries as:
        # class SearchQuery(BaseModel):
        #     search_query: str
        query_str = query.search_query if isinstance(query, SearchQuery) else str(query)  # text query

        try:
            # get results from tavily asynchronously (in parallel) for each search query
            search_tasks.append(
                TAVILY_SEARCH.raw_results_async(
                    query=query_str,
                    max_results=num_results,
                    search_depth="advanced",
                    include_answer=False,
                    include_raw_content=include_raw_content
                )
            )
        except Exception as e:
            print(f"Error creating search task for query '{query_str}': {e}")
            continue

    # Execute all searches concurrently and await results
    try:
        if not search_tasks:
            return []
        search_docs = await asyncio.gather(*search_tasks, return_exceptions=True)
        # Filter out any exceptions from the results
        valid_results = [
            doc for doc in search_docs
            if not isinstance(doc, Exception)
        ]
        return valid_results
    except Exception as e:
        print(f"Error during search queries: {e}")
        return []


def format_search_query_results(
        search_response: Union[Dict[str, Any], List[Any]],
        max_tokens: int = 2000,
        include_raw_content: bool = False,
        llm_model_name: str = "gpt-4"
) -> str:
    encoding = tiktoken.encoding_for_model(llm_model_name)
    sources_list = []

    # Handle different response formats
    # if search results is a dict
    if isinstance(search_response, dict):
        if "results" in search_response:
            sources_list.extend(search_response["results"])
        else:
            sources_list.append(search_response)
    # if search results is a list
    elif isinstance(search_response, list):
        for response in search_response:
            if isinstance(response, dict):
                if "results" in response:
                    sources_list.extend(response["results"])
                else:
                    sources_list.append(response)
            elif isinstance(response, list):
                sources_list.extend(response)

    if not sources_list:
        return "No search results found."

    # Deduplicate by URL and keep unique sources (website urls)
    unique_sources = {}
    for source in sources_list:
        if isinstance(source, dict) and 'url' in source:
            if source["url"] not in unique_sources:
                unique_sources[source['url']] = source

    # Format output
    formatted_text = "Content from web search:\n\n"
    for i, source in enumerate(unique_sources.values(), 1):
        formatted_text += f"Source {source.get('title', 'Untitled')}:\n===\n"
        formatted_text += f"URL: {source['url']}\n===\n"
        formatted_text += f"Most relevant content from source: {source.get('content', 'No content available')}\n===\n"

        if include_raw_content:
            # truncate raw webpage content to a certain number of tokens to prevent exceeding LLM max token window
            raw_content = source.get("raw_content", "")
            if raw_content:
                tokens = encoding.encode(raw_content)
                truncated_tokens = tokens[:max_tokens]
                truncated_content = encoding.decode(truncated_tokens)
                formatted_text += f"Raw Content: {truncated_content}\n\n"

    return formatted_text.strip()
