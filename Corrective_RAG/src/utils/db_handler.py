from pymilvus import connections, MilvusClient, CollectionSchema, FieldSchema, DataType
from typing import List


class MilvusSchemaBuilder:
    """Responsible for building collection schemas."""

    @staticmethod
    def build_schema(dimension: int) -> CollectionSchema:
        return CollectionSchema(
            fields=[
                FieldSchema(
                    name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
                ),
                FieldSchema(
                    name="content", dtype=DataType.VARCHAR, max_length=65535
                ),
                FieldSchema(name="metadata", dtype=DataType.JSON),
                FieldSchema(
                    name="vector", dtype=DataType.FLOAT_VECTOR, dim=dimension
                ),
            ],
            description="Schema for vector database storage",
            enable_dynamic_field=True,
        )


class MilvusManager:
    """Handles Milvus connection, database, and collection management."""

    def __init__(self, host: str = "localhost", port: str = "19530"):
        self.host = host
        self.port = port
        self.client = MilvusClient(uri=f"http://{host}:{port}")
        connections.connect(alias="default", host=host, port=port)

    def ensure_database(self, database_name: str) -> None:
        """Ensure database exists; create if missing."""
        if database_name not in self.client.list_databases():
            self.client.create_database(database_name)
            print(f"✅ Created database: {database_name}")
        else:
            print(f"ℹ️ Database {database_name} already exists")

        self.client.using_database(database_name)

    def ensure_collection(
        self,
        collection_name: str,
        schema: CollectionSchema,
        shards_num: int = 2,
    ) -> None:
        """Ensure collection exists; create if missing."""
        if collection_name not in self.client.list_collections():
            self.client.create_collection(
                collection_name=collection_name,
                schema=schema,
                shards_num=shards_num,
            )
            print(f"✅ Created collection: {collection_name}")
        else:
            print(f"ℹ️ Collection {collection_name} already exists")

    def get_client(self) -> MilvusClient:
        return self.client


def create_milvus_database_and_collection(
    database_name: str,
    collection_name: str,
    host: str = "localhost",
    port: str = "19530",
    dimension: int = 384,  # e.g., multilingual-e5-small
) -> MilvusClient:
    """
    Ensure Milvus database and collection exist and return a connected client.
    """
    try:
        manager = MilvusManager(host=host, port=port)
        manager.ensure_database(database_name)
        schema = MilvusSchemaBuilder.build_schema(dimension)
        manager.ensure_collection(collection_name, schema)
        return manager.get_client()
    except Exception as e:
        print(f"❌ Failed to create database/collection: {str(e)}")
        raise
