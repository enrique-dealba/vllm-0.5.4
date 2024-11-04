import logging
from typing import Any, Dict, Tuple, Union

import httpx
from vllm import SamplingParams

from app.config import settings
from app.langchain_structured_outputs import generate_structured_response
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


async def call_peer_service(query: str) -> Dict:
    """Call the other LLM service."""
    async with httpx.AsyncClient() as client:
        response = await client.post(
            f"{settings.PEER_SERVICE_URL}/generate", json={"text": query}
        )
        return response.json()


async def generate_response(
    user_input: str, call_peer: bool = False
) -> Tuple[Union[str, Dict[str, Any]], float]:
    """Generate response from LLM with optional peer service call."""
    try:
        # Get primary response
        if settings.USE_STRUCTURED_OUTPUT:
            response, execution_time = generate_structured_response(user_input)
            log_to_langsmith(
                chain_name="Structured Output Chain",
                inputs={"query": user_input},
                outputs={"response": response.model_dump()},
                metadata={
                    "model_type": settings.MODEL_TYPE,
                    "structured": True,
                    "service": settings.SERVICE_NAME,
                },
            )
        else:
            response, execution_time = generate_unstructured_response(user_input)
            log_to_langsmith(
                chain_name="Unstructured Output Chain",
                inputs={"query": user_input},
                outputs={"response": response},
                metadata={
                    "model_type": settings.MODEL_TYPE,
                    "structured": False,
                    "service": settings.SERVICE_NAME,
                },
            )

        # Get peer response if requested
        if call_peer and settings.PEER_SERVICE_URL:
            try:
                peer_response = await call_peer_service(user_input)

                # For structured output
                if settings.USE_STRUCTURED_OUTPUT:
                    combined_response = response.model_copy()
                    if hasattr(response, "evidence") and isinstance(
                        response.evidence, list
                    ):
                        # Add peer evidence with prefix
                        peer_evidence = peer_response.get("evidence", [])
                        combined_response.evidence = [
                            *response.evidence,
                            *[f"PEER: {e}" for e in peer_evidence],
                        ]

                    # Average confidences if available
                    if (
                        hasattr(response, "confidence")
                        and "confidence" in peer_response
                    ):
                        combined_response.confidence = (
                            response.confidence + peer_response["confidence"]
                        ) / 2

                    # Combine responses
                    combined_response.response = (
                        f"Primary: {response.response}\n"
                        f"Peer: {peer_response.get('response', 'No peer response')}"
                    )

                    response = combined_response

                # For unstructured output
                else:
                    response = (
                        f"Primary: {response}\n"
                        f"Peer: {peer_response.get('response', 'No peer response')}"
                    )

                # Log the combined response
                log_to_langsmith(
                    chain_name="Combined Output Chain",
                    inputs={"query": user_input},
                    outputs={
                        "primary_response": response,
                        "peer_response": peer_response,
                    },
                    metadata={
                        "model_type": settings.MODEL_TYPE,
                        "structured": settings.USE_STRUCTURED_OUTPUT,
                        "service": settings.SERVICE_NAME,
                        "peer_called": True,
                    },
                )

            except Exception as peer_error:
                logger.warning(
                    f"Peer service call failed: {peer_error}. "
                    "Proceeding with primary response only."
                )

        return response, execution_time

    except Exception as e:
        logger.exception(f"Error during LLM generation: {e}")
        raise
