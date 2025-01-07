import argparse
import yaml

from Simple_Rag.src.graph import create_graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", default=None, type=str,
                        help="Config file path (default: None)")
    args = parser.parse_args()

    if args.config is None:
        raise ValueError("The config argument should be set!")

    config = yaml.safe_load(open(args.config))

    GRAPH = create_graph(config)
