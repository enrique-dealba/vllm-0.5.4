from typing import List, Optional

from pydantic import BaseModel, Field


class BasicLLMResponse(BaseModel):
    response: str = Field(..., description="The main response from the LLM")


class DetailedLLMResponse(BaseModel):
    response: str = Field(..., description="The main response from the LLM")
    sources: Optional[List[str]] = Field(
        None, description="Sources or references for the response"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score of the response"
    )
