import argparse
import yaml

from Simple_Rag.src.graph import create_graph
from Simple_Rag.src.vector_store import MilvusHandler
from Simple_Rag.src.data_loader import get_data_loader, get_chunker


def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path: The path to the configuration file.

    Returns:
        The configuration as a dictionary.
    """
    with open(config_path, "r") as config_file:
        return yaml.safe_load(config_file)


def process_data(file_path: str, config: dict) -> list:
    """
    Load and chunk data for vector storage.

    Args:
        file_path: Path to the knowledge base file.
        config: Configuration for data loading and chunking.

    Returns:
        A list of chunked documents ready for embedding.
    """
    data_loader = get_data_loader()
    loaded_data = data_loader(file_path=file_path)
    chunker = get_chunker(chunker_config=config["chunker"])
    return chunker.split_documents(loaded_data)


def initialize_vector_db(config: dict, chunks: list) -> None:
    """
    Initialize the vector database and insert document chunks.

    Args:
        config: Configuration for the vector database and embedding model.
        chunks: The list of document chunks to insert into the vector DB.

    Returns:
        None
    """
    vector_db = MilvusHandler(vector_db_config=config["retriever"]["vector_db"],
                              retriever_model_config=config["retriever"]["embedding_model"])
    vector_db.create_collection()
    vector_db.insert_documents(chunks)


def main():
    """
    The main entry point for the script.
    """
    parser = argparse.ArgumentParser(description="Process and query knowledgebase data.")
    parser.add_argument(
        "-c", "--config", required=True, type=str, help="Config file path"
    )
    parser.add_argument(
        "-f", "--file_path", default=None, type=str, help="Knowledgebase file path"
    )
    parser.add_argument(
        "-l", "--is_loading", default=True, help="Load data into vector DB"
    )
    args = parser.parse_args()

    # Load the configuration
    config = load_config(args.config)

    # Process and load data into the vector database if specified
    if args.is_loading:
        chunks = process_data(args.file_path, config)
        initialize_vector_db(config, chunks)

    # Create and invoke the graph for querying
    graph = create_graph(config)

    while True:
        question = input("Enter your query: ")  # Reasons for and objectives of the proposal
        response = graph.invoke({"question": question})
        print(response["context"])
        print(response["answer"].content)


if __name__ == "__main__":
    main()
