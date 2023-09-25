import uvicorn
import os
from dotenv import load_dotenv
from fastapi import FastAPI
from app.api.routes import router
from app.resources.database import Database
from app.resources.LLM import LLM, LLMType

app = FastAPI()
app.include_router(router)

load_dotenv()
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
LLM_TYPE = os.getenv("LLM_TYPE")


@app.on_event("startup")
def startup_db_client():
    db = Database()
    llm = LLM(llm_type=LLMType(LLM_TYPE), hf_token=HF_TOKEN)
    app.db = db
    app.llm = llm


@app.on_event("shutdown")
def shutdown_db_client():
    pass


if __name__ == "__main__":
    uvicorn.run("main:app", reload=True)
