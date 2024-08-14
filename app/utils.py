from io import BytesIO

import requests
from PIL import Image

from app.config import settings


def load_image(url: str = settings.FIXED_IMAGE_URL) -> Image.Image:
    response = requests.get(url, timeout=settings.IMAGE_FETCH_TIMEOUT)
    return Image.open(BytesIO(response.content))
