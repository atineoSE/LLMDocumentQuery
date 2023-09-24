import uuid
from langchain.vectorstores import Chroma
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader
from langchain.schema import Document
import logging
import random


SPLIT_CHUNK_SIZE = 1500
SPLIT_CHUNK_OVERLAP = 150
SENTENCE_TRANSFORMERS_MODEL = "all-MiniLM-L6-v2"
DOCS_FOLDER = "docs"
MMR_K = 3
MMR_FETCH_K = 10


class Database:
    def __init__(self):
        self.persist_directory = f'{DOCS_FOLDER}/{str(uuid.uuid4())}/'
        self.embedding_function = SentenceTransformerEmbeddings(
            model_name=SENTENCE_TRANSFORMERS_MODEL)
        self.vectordb = Chroma(
            persist_directory=self.persist_directory,
            embedding_function=self.embedding_function
        )
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=SPLIT_CHUNK_SIZE,
            chunk_overlap=SPLIT_CHUNK_OVERLAP
        )

    def store(self, document_file: file) -> None:
        self.document_file = file
        logging.debug(
            "DATABASE: storing file at {self.document_file.__path__} in vector DB.")
        loader = PyPDFLoader(self.document_file.__path__)
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

        texts = [d.page_content for d in documents]
        metadatas = [d.metadata for d in documents]
        self.vectordb.add_texts(texts=texts, metadatas=metadatas)

    def retrieve(self, query: str) -> list[str] | None:
        if self.vectordb._collection.count() == 0:
            logging.debug(
                "DATABASE: no documents in the DB. Nothing to fetch.")
            return None
        documents = self.vectordb.max_marginal_relevance_search(
            query=query, k=MMR_K, fetch_k=MMR_FETCH_K)
        logging.debug(f"DATABASE: retrieved {len(documents)} matches")
        for document in documents:
            logging.debug(document.page_content)
            logging.debug(document.metadata)
            logging.debug("---")
        return [d.page_content for d in documents]

    def __del__(self):
        # TODO: cleanup resources
        # Remove file
        # Remove persist directory
        pass
