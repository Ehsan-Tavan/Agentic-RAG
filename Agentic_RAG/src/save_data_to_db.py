import yaml
import argparse

from retriver import create_milvus_database_and_collection, save_data_to_milvus
from utils import load_embedding_model, get_embedding_size, load_documents_from_json


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
    KNOWLEDGE_BASE = load_documents_from_json(input_file_path=CONFIG["data"]["knowledgebase_base_file"])
    EMBEDDING_MODEL = load_embedding_model(model_name=CONFIG["embedding_model"]["model_path"])
    EMBEDDING_SIZE = get_embedding_size(EMBEDDING_MODEL)

    create_milvus_database_and_collection(
        database_name=CONFIG["database"]["db_name"],
        collection_name=CONFIG["database"]["collection_name"],
        host=CONFIG["database"]["host"],
        port=CONFIG["database"]["port"],
        dimension=EMBEDDING_SIZE)

    save_data_to_milvus(documents=KNOWLEDGE_BASE,
                        db_name=CONFIG["database"]["db_name"],
                        collection_name=CONFIG["database"]["collection_name"],
                        host=CONFIG["database"]["host"],
                        port=CONFIG["database"]["port"],
                        embedding_model=EMBEDDING_MODEL,
                        chunk_size=CONFIG["chunker"]["chunk_size"],
                        chunk_overlap=CONFIG["chunker"]["chunk_overlap"])
