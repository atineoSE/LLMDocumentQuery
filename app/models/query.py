from enum import Enum
from pydantic import BaseModel, Field


class RetrieveStrategy(Enum):
    MMR = "MMR"
    SIMILAR = "SIMILAR"


class Query(BaseModel):
    text: str
    retrieve_strategy: RetrieveStrategy = Field(
        default=RetrieveStrategy.SIMILAR)
