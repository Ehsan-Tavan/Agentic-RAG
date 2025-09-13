import os
import yaml
import argparse

from langchain_core.runnables import RunnableConfig
from uuid import uuid4

from Corrective_RAG.src.graph import create_graph

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
    os.environ["TAVILY_API_KEY"] = CONFIG["llm"]["api_key"]

    GRAPH = create_graph(CONFIG)

    # Configure settings (maximum recursion limit, thread_id)
    config = RunnableConfig(recursion_limit=20, configurable={"thread_id": uuid4()})

    # Input question
    inputs = {
        "question": "Who is Donald Trump?",
    }
    # Execute the graph in update format.
    for output in GRAPH.stream(inputs, config, stream_mode="updates"):
        for key, value in output.items():
            print(f"Output from node '{key}':")
            print("---")
            # print(value)
        print("\n---\n")

    print(value["generation"])
