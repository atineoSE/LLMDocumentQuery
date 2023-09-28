import os
from enum import Enum
import logging
from huggingface_hub import login
from dotenv import load_dotenv
import torch
from transformers import AutoTokenizer, pipeline
from transformers.tokenization_utils import PreTrainedTokenizerBase
from transformers.pipelines import Pipeline
from app.resources.prompts import llm_prompt


class LLMType(Enum):
    LLAMA2 = "LLAMA2"
    FALCON = "FALCON"
    MPT = "MPT"
    FAKE = "FAKE"

    @property
    def model_full_name(self):
        match self:
            case LLMType.LLAMA2:
                return "meta-llama/Llama-2-7b-chat-hf"
            case LLMType.FALCON:
                return "tiiuae/falcon-7b-instruct"
            case LLMType.MPT:
                return "mosaicml/mpt-7b-instruct"


class LLM:
    llm_type: LLMType
    tokenizer: PreTrainedTokenizerBase
    pipeline: Pipeline

    def __init__(self, llm_type: LLMType, hf_token: str | None = None):
        self.llm_type = llm_type

        if llm_type == LLMType.FAKE:
            logging.debug("LLM: started FAKE LLM")
            return

        if hf_token:
            login(token=hf_token)

        # Initialize model
        self.tokenizer = AutoTokenizer.from_pretrained(
            llm_type.model_full_name)
        self.pipeline = pipeline(
            "text-generation",
            tokenizer=self.tokenizer,
            model=llm_type.model_full_name,
            torch_dtype=torch.float16,
            device_map="auto",
        )
        logging.debug(f"LLM: started {llm_type.name} LLM")

    def predict(self, query: str, texts: list[str]) -> str:
        if self.llm_type == LLMType.FAKE:
            return f"As a fake LLM, I can repeat your question \"{query.text}\" and quote the first text:\n{texts[0]}"

        prompt = llm_prompt.format(
            context="\n---\n".join(texts),
            query=query
        )
        logging.debug(f"LLM: calling model {self.llm_type.name} with prompt:")
        logging.debug(prompt)

        result = self.pipeline(
            prompt,
            eos_token_id=self.tokenizer.eos_token_id,
            max_length=2000
        )
        logging.debug(f"LLM: got results {result}")

        answer = result[0]["generated_text"].replace(prompt, "").strip()
        split_answer = answer.split("\n")[0]
        logging.debug(f"LLM: answer is \"{split_answer}\"")

        return split_answer
