import os
import yaml
import argparse

from Reflection_Agent.src.graph import create_graph


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

    graph = create_graph(CONFIG)

    # Input question
    inputs = {
        "user_request": "Write a Python function to find the nth Fibonacci number.",
    }

    a = graph.invoke(inputs)
    print(a["user_request"])
    print(a["draft"].code)
    print(a["draft"].explanation)