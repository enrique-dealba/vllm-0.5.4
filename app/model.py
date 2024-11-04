import logging
import os

import torch
from langchain_community.llms import VLLM as LangChainVLLM
from vllm import LLM as VLM

from app.config import settings
from app.utils import load_image

logger = logging.getLogger(__name__)

llm = None
vlm = None
image = None


def check_gpu_status():
    """Check and log GPU status."""
    try:
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        logger.info(f"Number of GPUs: {torch.cuda.device_count()}")
        logger.info(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

        # Check each GPU
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            mem_used = torch.cuda.memory_allocated(i) / 1024**3  # Convert to GB
            mem_total = props.total_memory / 1024**3
            logger.info(
                f"GPU {i}: {props.name}, Memory: {mem_used:.2f}GB / {mem_total:.2f}GB"
            )

    except Exception as e:
        logger.error(f"Error checking GPU status: {e}")


def initialize_models():
    global llm, vlm, image

    # Log initial state
    logger.info(
        f"Initializing {settings.MODEL_TYPE} for service {settings.SERVICE_NAME}"
    )
    logger.info(f"Target CUDA device: {settings.CUDA_DEVICE}")
    check_gpu_status()

    # Set CUDA device first
    target_device = str(settings.CUDA_DEVICE)
    os.environ["CUDA_VISIBLE_DEVICES"] = target_device
    logger.info(f"Set CUDA_VISIBLE_DEVICES={target_device}")

    # Verify CUDA setup after setting device
    check_gpu_status()

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
            llm = LangChainVLLM(
                model=settings.LLM_MODEL_NAME,
                trust_remote_code=True,  # Mandatory for Hugging Face models
                max_new_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                tokenizer_mode=tokenizer_mode,
                vllm_kwargs={
                    "tokenizer_mode": tokenizer_mode,
                    # "gpu_memory_utilization": gpu_utilization,
                },
            )
            logger.info(
                f"LangChain LLM model '{settings.LLM_MODEL_NAME}' initialized successfully "
                f"on CUDA device {settings.CUDA_DEVICE}"
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


# Initialize models when module is loaded
try:
    initialize_models()
except Exception as e:
    logger.error(f"Failed to initialize models on startup: {e}")
    # Don't raise here - let the health check endpoint handle it
