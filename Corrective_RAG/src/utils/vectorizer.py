from abc import ABC, abstractmethod
from typing import List
from tqdm import tqdm
from langchain_community.document_loaders import PDFPlumberLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient


class DocumentLoader(ABC):
    @abstractmethod
    def load(self, doc_paths: List[str]) -> List[Document]:
        pass


class PDFDocumentLoader(DocumentLoader):
    def load(self, doc_paths: List[str]) -> List[Document]:
        loaded_docs = []
        for doc_path in doc_paths:
            loader = PDFPlumberLoader(doc_path)
            loaded_docs.extend(loader.load())
        return loaded_docs


class EmbeddingModel(ABC):
    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        pass


class SentenceTransformerEmbeddingModel(EmbeddingModel):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformerEmbeddings(model_name=model_name)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)


class OpenAIEmbeddingModel:
    def __init__(self):
        self.model = OpenAIEmbeddings()

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.embed_documents(texts)


class VectorDBClient(ABC):
    """Abstract base class for Vector DB clients."""

    @abstractmethod
    def insert(self, collection_name: str, data: List[dict]) -> None:
        pass


class MilvusVectorDBClient(VectorDBClient):
    def __init__(self, uri: str, db_name: str):
        self.client = MilvusClient(uri=uri, db_name=db_name)

    def insert(self, collection_name: str, data: List[dict]) -> None:
        self.client.insert(collection_name=collection_name, data=data)


class Vectorizer:
    def __init__(
            self,
            source_uri: List[str],
            document_loader: DocumentLoader,
            embedding_model: EmbeddingModel,
            vector_db_client: VectorDBClient,
            chunk_size: int = 300,
            chunk_overlap: int = 50

    ):
        self.source_uri = source_uri
        self.document_loader = document_loader
        self.embedding_model = embedding_model
        self.vector_db_client = vector_db_client
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def create_text_splitter(self):
        return RecursiveCharacterTextSplitter(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)

    def process_documents(self) -> List[Document]:
        documents = self.document_loader.load(self.source_uri)
        text_splitter = self.create_text_splitter()
        return text_splitter.split_documents(documents)

    def insert_to_vector_db(self, collection_name: str, batch_size: int = 32) -> None:
        try:
            docs = self.process_documents()
            texts = [doc.page_content for doc in docs]
            metadatas = [doc.metadata for doc in docs]

            total = len(texts)
            print(f"📄 Total documents to insert: {total}")

            for i in tqdm(range(0, total, batch_size), desc="Embedding & Inserting", unit="batch"):
                batch_texts = texts[i:i + batch_size]
                batch_metadatas = metadatas[i:i + batch_size]

                batch_embeddings = self.embedding_model.embed_documents(batch_texts)

                batch_data = [
                    {"content": text, "metadata": metadata, "vector": embedding}
                    for text, metadata, embedding in zip(batch_texts, batch_metadatas, batch_embeddings)
                ]

                self.vector_db_client.insert(collection_name=collection_name, data=batch_data)

            print(f"✅ Inserted {total} documents into collection {collection_name}")

        except Exception as e:
            print(f"Failed to save data to Milvus: {str(e)}")
            raise
