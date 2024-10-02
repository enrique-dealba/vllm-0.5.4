import time
from typing import List, Optional

from langchain_core.runnables import RunnablePassthrough
from pydantic import BaseModel, Field

from app.model import llm


class LLMResponse(BaseModel):
    response: str = Field(..., description="The main response from the LLM")
    sources: Optional[List[str]] = Field(
        None, description="Sources or references for the response"
    )
    confidence: float = Field(
        ..., ge=0, le=1, description="Confidence score of the response"
    )


def generate_response(user_input: str) -> tuple[LLMResponse, float]:
    start_time = time.time()

    try:
        structured_llm = llm.with_structured_output(LLMResponse, method="json_schema")

        chain = RunnablePassthrough.assign(structured_output=structured_llm) | (
            lambda x: x["structured_output"]
        )

        response = chain.invoke(user_input)

        end_time = time.time()
        execution_time = end_time - start_time

        return response, execution_time

    except Exception as e:
        print(f"Error during structured output generation: {e}")
        raise
