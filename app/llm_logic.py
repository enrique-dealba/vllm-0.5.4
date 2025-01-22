import logging
from typing import Any, Dict, Union

from vllm import SamplingParams

from app.config import settings
from app.langchain_structured_outputs import generate_objective_response
from app.langchain_structured_outputs import generate_structured_response as generate
from app.model import image, llm, vlm
from app.utils import log_to_langsmith, time_function

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@time_function
def generate_unstructured_response(user_input: str) -> str:
    if settings.MODEL_TYPE.upper() == "LLM":
        if llm is None:
            raise ValueError("LLM is not available.")
        return llm.invoke(user_input)
    elif settings.MODEL_TYPE.upper() == "VLM":
        if vlm is None:
            raise ValueError("VLM is not available.")
        prompt = f"USER: <image>\n{user_input}\nASSISTANT:"
        sampling_params = SamplingParams(
            temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS
        )
        outputs = vlm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            ],
            sampling_params,
        )
        return outputs[0].outputs[0].text
    else:
        raise ValueError("Invalid MODEL_TYPE configuration.")


def generate_response(user_input: str) -> Union[str, Dict[str, Any]]:
    try:
        if settings.USE_STRUCTURED_OUTPUT:
            response, execution_time = generate(user_input)
            log_to_langsmith(
                chain_name="Structured Output Chain",
                inputs={"query": user_input},
                outputs={"response": response.model_dump()},
                metadata={"model_type": settings.MODEL_TYPE, "structured": True},
            )
        else:
            response, execution_time = generate_unstructured_response(user_input)
            log_to_langsmith(
                chain_name="Unstructured Output Chain",
                inputs={"query": user_input},
                outputs={"response": response},
                metadata={"model_type": settings.MODEL_TYPE, "structured": False},
            )

        return response, execution_time
    except Exception as e:
        logger.exception(f"Error during LLM generation: {e}")
        raise


def generate_objective(user_input: str) -> Union[str, Dict[str, Any]]:
    try:
        assert settings.USE_STRUCTURED_OUTPUT
        response, execution_time = generate_objective_response(user_input)
        log_to_langsmith(
            chain_name="Structured Output Chain",
            inputs={"query": user_input},
            outputs={"response": response.model_dump()},
            metadata={"model_type": settings.MODEL_TYPE, "structured": True},
        )
        return response, execution_time
    except Exception as e:
        logger.exception(f"Error during LLM generation: {e}")
        raise
