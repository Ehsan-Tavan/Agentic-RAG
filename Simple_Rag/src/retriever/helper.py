from langchain_milvus import Milvus


def get_milvus_collection(embedding_function, connection_args, collection_name):
    milvus_collection = Milvus(
        embedding_function=embedding_function,
        connection_args=connection_args,
        collection_name=collection_name,
    )
    return milvus_collection
