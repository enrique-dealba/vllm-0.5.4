import time
from typing import List, Optional

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
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
        parser = PydanticOutputParser(pydantic_object=LLMResponse)

        prompt = PromptTemplate(
            template="Answer the user query.\n{format_instructions}\n{query}\n",
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | llm | parser

        response = chain.invoke({"query": user_input})

        end_time = time.time()
        execution_time = end_time - start_time

        return response, execution_time

    except Exception as e:
        print(f"Error during structured output generation: {e}")
        raise
