import os
import logging
import sys
from dotenv import load_dotenv
from app.api.routes import router
from app.resources.database import Database
from app.resources.LLM import LLM, LLMType

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LLM_TYPE = os.getenv("LLM_TYPE")

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    stream=sys.stdout
)


class State:
    db: Database
    llm: LLM

    def __init__(self):
        self.db = Database()
        self.llm = LLM(llm_type=LLMType(LLM_TYPE), hf_token=HF_TOKEN)
