import logging
import random
import uuid
import shutil
import os
from io import IOBase
from pathlib import Path
from typing import Mapping, Any, BinaryIO
from chromadb import chromadb
from chromadb.api import API
from chromadb.config import Settings
from chromadb.api.models.Collection import Collection
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
from app.models.query import Query, RetrieveStrategy

SPLIT_CHUNK_SIZE = 1500
SPLIT_CHUNK_OVERLAP = 150
SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
DB_FOLDER = "db"
FILES_FOLDER = "files"
COLLECTION_NAME = "LLM_VECTOR_DB"
TOP_K_MATCHES = 3
MMR_FETCH_K = 10


class Database:
    client: API
    collection: Collection
    vector_db: Chroma
    embedding_function: SentenceTransformerEmbeddings
    text_splitter: RecursiveCharacterTextSplitter

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

    def store(self, document: BinaryIO) -> None:
        self._cleanup_previous_document()

        # Persist file locally
        document_path = f"{FILES_FOLDER}/{uuid.uuid4()}.pdf"
        with open(document_path, "wb") as file:
            shutil.copyfileobj(document, file)

        # Load and split document into chunks
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

        # Store chunks in DB
        ids = [str(x) for x in range(len(documents))]
        metadatas: list[Mapping[str, Any]] = [d.metadata for d in documents]
        texts = [d.page_content for d in documents]
        self.collection.add(
            ids=ids,
            metadatas=metadatas,
            documents=texts
        )

    def retrieve(self, query: Query) -> list[str]:
        if self.client.get_or_create_collection(name=COLLECTION_NAME).count() == 0:
            logging.debug(
                "DATABASE: no documents in the DB. Nothing to fetch.")
            return []

        documents: list[Document] = []
        logging.debug(
            f"DATABASE: retrieving documents for query \"{query.text}\" with strategy {query.retrieve_strategy.name}")
        match query.retrieve_strategy:
            case RetrieveStrategy.MMR:
                documents = self.vector_db.max_marginal_relevance_search(
                    query=query.text, k=TOP_K_MATCHES, fetch_k=MMR_FETCH_K
                )
            case RetrieveStrategy.SIMILAR:
                documents = self.vector_db.similarity_search(
                    query=query.text, k=TOP_K_MATCHES
                )

        logging.debug(f"DATABASE: retrieved {len(documents)} matches")
        for document in documents:
            logging.debug(document.page_content)
            logging.debug(document.metadata)
            logging.debug("---")
        return [d.page_content for d in documents]

    def _cleanup_previous_document(self):
        # Cleanup previous files
        for path in Path(FILES_FOLDER).glob("*.pdf"):
            os.remove(path)

        # Cleanup DB
        self.client.delete_collection(name=COLLECTION_NAME)
        self.collection = self.client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=self.embedding_function.embed_documents
        )
        logging.debug("DATABASE: cleaned up vector DB")
