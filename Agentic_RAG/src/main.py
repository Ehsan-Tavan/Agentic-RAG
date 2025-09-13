import os
import yaml
import argparse

from graph import create_graph



if __name__ == "__main__":
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


    GRAPH = create_graph(config=CONFIG)

    for chunk in GRAPH.stream(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "What does Lilian Weng say about types of reward hacking?",
                    }
                ]
            }
    ):
        for node, update in chunk.items():
            print("Update from node", node)
            # print(update["messages"][-1])
            print("\n\n")



