import logging
from typing import Any, AsyncGenerator, Dict, Union

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from vllm import SamplingParams

from app.config import settings
from app.model import image, llm, vlm
from app.utils import load_schema, log_to_langsmith, time_function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@time_function
async def generate_unstructured_response_stream(
    user_input: str,
) -> AsyncGenerator[str, None]:
    if settings.MODEL_TYPE.upper() == "LLM":
        if llm is None:
            raise ValueError("LLM is not available.")
        async for chunk in llm.astream(user_input):
            yield chunk.content
    elif settings.MODEL_TYPE.upper() == "VLM":
        if vlm is None:
            raise ValueError("VLM is not available.")
        prompt = f"USER: <image>\n{user_input}\nASSISTANT:"
        sampling_params = SamplingParams(
            temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS
        )
        async for output in vlm.astream_generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            ],
            sampling_params,
        ):
            yield output.text
    else:
        raise ValueError("Invalid MODEL_TYPE configuration.")


@time_function
async def generate_structured_response_stream(
    user_input: str,
) -> AsyncGenerator[Dict[str, Any], None]:
    LLMResponseSchema = load_schema()

    parser = PydanticOutputParser(pydantic_object=LLMResponseSchema)

    prompt = PromptTemplate(
        template="Answer the user query.\n{format_instructions}\n{query}\n",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | llm | parser

    async for chunk in chain.astream({"query": user_input}):
        # Assuming chunk is a partial output of the schema
        yield chunk.model_dump()


@time_function
def generate_stream_response(user_input: str) -> Union[AsyncGenerator, Dict[str, Any]]:
    try:
        if settings.USE_STRUCTURED_OUTPUT:
            response_generator, execution_time = generate_structured_response_stream(
                user_input
            )
            log_to_langsmith(
                chain_name="Structured Output Stream Chain",
                inputs={"query": user_input},
                outputs={"response": "Streaming..."},
                metadata={"model_type": settings.MODEL_TYPE, "structured": True},
            )
        else:
            response_generator, execution_time = generate_unstructured_response_stream(
                user_input
            )
            log_to_langsmith(
                chain_name="Unstructured Output Stream Chain",
                inputs={"query": user_input},
                outputs={"response": "Streaming..."},
                metadata={"model_type": settings.MODEL_TYPE, "structured": False},
            )

        return response_generator, execution_time
    except Exception as e:
        logger.exception(f"Error during streaming LLM generation: {e}")
        raise
