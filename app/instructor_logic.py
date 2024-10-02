import time
from typing import List, Optional

import instructor
from langchain_community.chat_models import ChatOpenAI
from pydantic import BaseModel, Field

from app.config import settings


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
        model = ChatOpenAI(
            openai_api_base=settings.OPENAI_API_BASE,
            openai_api_key=settings.OPENAI_API_KEY,
            model_name=settings.LLM_MODEL_NAME,
            temperature=settings.TEMPERATURE,
            max_tokens=settings.MAX_TOKENS,
        )

        client = instructor.patch(model)

        response = client.chat.completions.create(
            model=settings.LLM_MODEL_NAME,
            response_model=LLMResponse,
            messages=[
                {"role": "system", "content": "You are a helpful AI assistant."},
                {"role": "user", "content": user_input},
            ],
        )

        end_time = time.time()
        execution_time = end_time - start_time

        return response, execution_time

    except Exception as e:
        print(f"Error during generation: {e}")
        raise
