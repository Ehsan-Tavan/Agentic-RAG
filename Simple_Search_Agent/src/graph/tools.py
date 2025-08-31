from langchain_tavily import TavilySearch


def get_tools(configs):
    search = TavilySearch(max_results=configs["num_search_results"])
    return [search]