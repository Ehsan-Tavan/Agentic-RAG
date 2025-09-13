import os
import yaml
import argparse

from langchain_openai import ChatOpenAI
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.prebuilt import create_react_agent
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langchain_core.messages import ToolMessage

from Simple_Search_Agent.src.graph import create_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deep Research AI Agent"
    )
    parser.add_argument(
        "-c", "--config",
        default=None,
        type=str,
        help="Config file path (default: None)"
    )

    args = parser.parse_args()

    # Validate required arguments
    if args.config is None:
        raise ValueError("The config argument should be set!")

    CONFIG = yaml.safe_load(open(args.config))

    os.environ["OPENAI_API_KEY"] = CONFIG["llm"]["api_key"]

    api_key = os.environ["TAVILY_API_KEY"] = CONFIG["search_tools"]["tavily_search_api_key"]

    graph = create_graph(CONFIG)


    def stream_graph_updates(user_input: str):
        for event in graph.stream({"messages": [{"role": "user", "content": user_input}]}):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)


    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break

            stream_graph_updates(user_input)
        except:
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
