import os
import yaml
import argparse

from Corrective_RAG.src.utils import (create_milvus_database_and_collection, PDFDocumentLoader,
                                      SentenceTransformerEmbeddingModel, MilvusVectorDBClient,
                                      Vectorizer)

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

    create_milvus_database_and_collection(
        database_name=CONFIG["database"]["db_name"],
        collection_name=CONFIG["database"]["collection_name"],
        host=CONFIG["database"]["host"],
        port=CONFIG["database"]["port"],
    )

    LOADER_OBJ = PDFDocumentLoader()
    EMBEDDING_OBJ = SentenceTransformerEmbeddingModel(model_name=CONFIG["retriever"]["embedding_model_path"])
    DB_CLIENT_OBJ = MilvusVectorDBClient(uri=f"http://{CONFIG['database']['host']}:{CONFIG['database']['port']}",
                                         db_name=CONFIG["database"]["db_name"])

    VECTORIZER = Vectorizer(source_uri=CONFIG["data"]["knowledgebase_base_file"],
                            document_loader=LOADER_OBJ,
                            embedding_model=EMBEDDING_OBJ,
                            vector_db_client=DB_CLIENT_OBJ)

    VECTORIZER.insert_to_vector_db(collection_name=CONFIG["database"]["collection_name"])


