import importlib
import time
from io import BytesIO
from typing import Any, Dict, Type
from uuid import uuid4

import requests
from langchain.callbacks.tracers import LangChainTracer
from PIL import Image
from pydantic import BaseModel

from app.config import settings

tracer = LangChainTracer(project_name=settings.LANGCHAIN_PROJECT)


def load_image(url: str = settings.FIXED_IMAGE_URL) -> Image.Image:
    try:
        response = requests.get(url, timeout=settings.IMAGE_FETCH_TIMEOUT)
        response.raise_for_status()
        return Image.open(BytesIO(response.content))
    except requests.RequestException as e:
        print(f"Failed to load image from {url}: {e}")
        raise


def load_schema() -> Type[BaseModel]:
    module = importlib.import_module("app.schemas.llm_responses")
    schema_class = getattr(module, settings.LLM_RESPONSE_SCHEMA)
    return schema_class


def log_to_langsmith(
    chain_name: str,
    inputs: Dict[str, Any],
    outputs: Dict[str, Any],
    metadata: Dict[str, Any],
):
    unique_id = str(uuid4())
    tracer.on_chain_start(
        {"name": chain_name},
        inputs,
        run_id=unique_id,
        tags=["streamlit_ui"],
        metadata=metadata,
    )
    tracer.on_chain_end(
        outputs=outputs,
        run_id=unique_id,
    )


def time_function(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time
        return result, execution_time

    return wrapper
