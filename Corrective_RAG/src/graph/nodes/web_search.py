import os
from requests.exceptions import HTTPError
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from langchain.schema import Document

from Corrective_RAG.src.graph.state import GraphState


class WebSearch:
    def __init__(self):
        self.web_search_tool = TavilySearchResults(max_results=3)

    def __call__(self, state: GraphState):
        print("\n==== [WEB SEARCH] ====\n")
        question = state["question"]
        documents = state["documents"]

        docs = []
        try:
            docs = self.web_search_tool.invoke({"query": question})

            # Additional check: Ensure docs is a list (handles cases where invoke returns a string error)
            if not isinstance(docs, list):
                print(f"Unexpected return type from invoke: {type(docs)}. Treating as failure.")
                docs = []

        except HTTPError as e:
            print(f"HTTPError occurred: {e}")
            docs = []
        except Exception as e:
            print(f"Unexpected error occurred: {e}")
            docs = []

        if docs and isinstance(docs, list):
            web_results = "\n".join([d["content"] for d in docs if isinstance(d, dict) and "content" in d])
            web_results = Document(page_content=web_results)
            documents.append(web_results)
        else:
            print("No valid search results to process.")

        return {"documents": documents}


def get_web_search_node():
    return WebSearch()
