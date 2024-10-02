import importlib
from io import BytesIO
from typing import Type

import requests
from PIL import Image
from pydantic import BaseModel

from app.config import settings


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
