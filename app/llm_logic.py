import logging
import time
from uuid import uuid4

from langchain.callbacks.tracers import LangChainTracer
from vllm import SamplingParams

from app.config import settings
from app.model import image, llm, vlm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize LangSmith tracing
tracer = LangChainTracer(project_name=settings.LANGCHAIN_PROJECT)


def generate_response(user_input):
    start_time = time.time()
    unique_id = str(uuid4())
    generated_text = ""

    try:
        if settings.MODEL_TYPE.upper() == "LLM":
            if llm is None:
                raise ValueError("LLM is not available.")
            generated_text = llm.invoke(user_input)
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
            generated_text = outputs[0].outputs[0].text
        else:
            raise ValueError("Invalid MODEL_TYPE configuration.")

        end_time = time.time()
        execution_time = end_time - start_time

        # Log the run to LangSmith
        tracer.on_chain_start(
            {"name": "Streamlit UI Chain"},
            {"query": user_input},
            run_id=unique_id,
            tags=["streamlit_ui"],
            metadata={
                "model_type": settings.MODEL_TYPE,
            },
        )
        tracer.on_chain_end(
            outputs={"response": generated_text},
            run_id=unique_id,
        )

        return generated_text, execution_time
    except Exception as e:
        logger.exception(f"Error during generation: {e}")
        raise
