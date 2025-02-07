import logging
import os

from langchain_community.llms import VLLM as LangChainVLLM
from vllm import LLM as VLM

from app.config import settings
from app.utils import load_image

logger = logging.getLogger(__name__)

llm = None
vlm = None
image = None


class MockLLM:
    def __init__(self):
        """Mock LLM for testing."""
        self.response_type = "mock"

    def invoke(self, text: str) -> dict:
        return {
            "objective_name": f"Mock-{text[::-1]}",  # Reversed user text
            "classification_marking": "U",
            "collect_request_type": "MOCK_REQUEST",
            "data_mode": "MOCK",
            "end_time_offset_minutes": 42,
            "final_offset": 24,
            "frame_overlap_percentage": 0.27,
            "frame_type": "MOCK",
            "initial_offset": 30,
            "objective_uuid": "mock-3141",
            "priority": 1000,
            "search_type": "MOCK_SEARCH",
            "sensor_name": "RME-MOCK",
            "target_id": "98765",
        }


def initialize_models():
    global llm, vlm, image

    if os.getenv("USE_MOCK_LLM", "false").lower() == "true":
        llm = MockLLM()
        logger.info("Initialized Mock LLM")
        return

    # Set Hugging Face Hub Token
    if settings.HUGGING_FACE_HUB_TOKEN:
        os.environ["HUGGINGFACEHUB_API_TOKEN"] = settings.HUGGING_FACE_HUB_TOKEN
        logger.info("Hugging Face Hub token set successfully.")
    else:
        logger.warning(
            "Hugging Face Hub token is not set. Proceeding without authentication."
        )

    # LangSmith setup
    os.environ["LANGCHAIN_TRACING_V2"] = str(settings.LANGCHAIN_TRACING_V2)
    os.environ["LANGCHAIN_ENDPOINT"] = settings.LANGCHAIN_ENDPOINT
    os.environ["LANGCHAIN_API_KEY"] = settings.LANGCHAIN_API_KEY
    os.environ["LANGCHAIN_PROJECT"] = settings.LANGCHAIN_PROJECT

    if settings.MODEL_TYPE.upper() == "LLM":
        try:
            tokenizer_mode = "mistral" if settings.IS_MISTRAL else "auto"
            load_format = settings.LOAD_FORMAT
            quantization = settings.QUANTIZATION
            llm = LangChainVLLM(
                model=settings.LLM_MODEL_NAME,
                trust_remote_code=True,  # Mandatory for Hugging Face models
                max_new_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                tokenizer_mode=tokenizer_mode,
                vllm_kwargs={
                    "tokenizer_mode": tokenizer_mode,
                    "load_format": load_format,
                    "quantization": quantization,
                    # "gpu_memory_utilization": gpu_utilization,
                },
            )
            logger.info(
                f"LangChain LLM model '{settings.LLM_MODEL_NAME}' initialized successfully."
            )
        except Exception as e:
            logger.error(f"Error initializing LangChain LLM: {e}")
            llm = None
    elif settings.MODEL_TYPE.upper() == "VLM":
        try:
            vlm = VLM(
                model=settings.VLM_MODEL_NAME,
                # Add additional VLM-specific parameters if needed
            )
            image = load_image(url=settings.FIXED_IMAGE_URL)
            logger.info(
                f"VLM model '{settings.VLM_MODEL_NAME}' initialized successfully with image '{settings.FIXED_IMAGE_URL}'."
            )
        except Exception as e:
            logger.error(f"Error initializing VLM: {e}")
            vlm = None
    else:
        logger.critical(
            f"Invalid MODEL_TYPE: {settings.MODEL_TYPE}. Must be 'LLM' or 'VLM'."
        )
        raise ValueError(
            f"Invalid MODEL_TYPE: {settings.MODEL_TYPE}. Must be 'LLM' or 'VLM'."
        )


# Initialize models
initialize_models()
