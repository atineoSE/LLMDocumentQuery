import logging
import random
from io import IOBase
from chromadb import chromadb
from chromadb.api import API
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document

SPLIT_CHUNK_SIZE = 1500
SPLIT_CHUNK_OVERLAP = 150
SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
DB_FOLDER = "db"
COLLECTION_NAME = "LLM_VECTOR_DB"
MMR_K = 3
MMR_FETCH_K = 10

File = IOBase


class Database:
    client: API
    collection: Collection
    vector_db: Chroma
    embedding_function: SentenceTransformerEmbeddings
    text_splitter: RecursiveCharacterTextSplitter
    document_path: str

    def __init__(self):
        self.client = chromadb.PersistentClient(
            path=DB_FOLDER,
            settings=Settings(anonymized_telemetry=False)
        )
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=SENTENCE_TRANSFORMERS_MODEL)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function.embed_documents
        )
        self.vector_db = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embedding_function,
            client=self.client
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SPLIT_CHUNK_SIZE,
            chunk_overlap=SPLIT_CHUNK_OVERLAP
        )
        self.document_path = None

    def store(self, document_path: str) -> None:
        self.cleanup()
        self.document_path = document_path
        loader = PyPDFLoader(document_path)
        documents = loader.load_and_split(text_splitter=self.text_splitter)

        logging.debug(
            f"DATABASE: split input document into {len(documents)} texts")
        try:
            samples = random.sample(documents, 2)
            logging.debug("DATABASE: Sample texts:")
            for sample in samples:
                logging.debug(sample.page_content)
        except ValueError:
            # Not enough text samples
            pass

        ids = [str(x) for x in range(len(documents))]
        metadatas = [d.metadata for d in documents]
        texts = [d.page_content for d in documents]
        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=texts
        )

    def retrieve(self, query: str) -> list[str]:
        if self.vector_db._collection.count() == 0:
            logging.debug(
                "DATABASE: no documents in the DB. Nothing to fetch.")
            return []
        documents = self.vector_db.max_marginal_relevance_search(
            query=query, k=MMR_K, fetch_k=MMR_FETCH_K)
        logging.debug(f"DATABASE: retrieved {len(documents)} matches")
        for document in documents:
            logging.debug(document.page_content)
            logging.debug(document.metadata)
            logging.debug("---")
        return [d.page_content for d in documents]

    def cleanup(self):
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function.embed_documents
        )
        logging.debug("DATABASE: cleaned up vector DB")
