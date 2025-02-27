from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    def update_schema(self, new_schema: str):
        """Update LLM response schema and notify observers"""
        self.LLM_RESPONSE_SCHEMA = new_schema
        if hasattr(self, "_schema_cache"):
            delattr(self, "_schema_cache")

    # General Settings
    PORT: int = 8888
    HOST: str = "0.0.0.0"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 8192  # prev: 256,
    MODEL_TYPE: str = "LLM"  # Default to LLM; options: 'LLM', 'VLM'
    TENSOR_PARALLEL_SIZE: int = 1
    GPU_UTILIZATION: float = 1.0

    # LLM Settings
    LLM_MODEL_NAME: str = "tiiuae/Falcon3-7B-Instruct"
    IS_MISTRAL: bool = False
    LOAD_FORMAT: str = "auto"
    QUANTIZATION: Optional[str] = None

    # VLM Settings
    VLM_MODEL_NAME: str = "llava-hf/llava-1.5-7b-hf"
    FIXED_IMAGE_URL: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    IMAGE_FETCH_TIMEOUT: int = 5

    # Hugging Face Settings
    HUGGING_FACE_HUB_TOKEN: str = ""

    # LangSmith Settings
    LANGCHAIN_TRACING_V2: bool = True  # prev: False
    LANGCHAIN_ENDPOINT: str = "https://api.smith.langchain.com"
    LANGCHAIN_API_KEY: str = ""
    LANGCHAIN_PROJECT: str = "demo-20241003"

    # LLM Structured Output Settings
    USE_STRUCTURED_OUTPUT: bool = True
    LLM_RESPONSE_SCHEMA: str = "BasicLLMResponse"


settings = Settings()
