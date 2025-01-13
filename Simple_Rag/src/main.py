import argparse
import yaml

from Simple_Rag.src.graph import create_graph
from Simple_Rag.src.vector_store import MilvusHandler
from Simple_Rag.src.data_loader import get_data_loader, get_chunker


def load_config(config_path):
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def process_data(file_path, config):
    data_loader = get_data_loader()
    loaded_data = data_loader(file_path=file_path)
    chunker = get_chunker(chunker_config=config["chunker"])
    return chunker.split_documents(loaded_data)


def initialize_vector_db(config, chunks):
    vector_db = MilvusHandler(config["vector_db"], config["retriever"]["embedding_model"])
    vector_db.create_collection()
    vector_db.insert_documents(chunks)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", required=True, type=str, help="Config file path")
    parser.add_argument("-f", "--file_path", default=None, type=str, help="Knowledgebase file path")
    parser.add_argument("-l", "--is_loading", action='store_true',
                        default=True, help="Load data into vector DB")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.is_loading:
        chunks = process_data(args.file_path, config)
        initialize_vector_db(config, chunks)

    graph = create_graph(config)
    response = graph.invoke({"question": "Reasons for and objectives of the proposal"})
    print(response["context"])
    print(response["answer"].content)


if __name__ == "__main__":
    main()
