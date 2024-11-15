from datetime import timedelta

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # General Settings
    PORT: int = 8888
    HOST: str = "0.0.0.0"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 8192  # prev: 256,
    MODEL_TYPE: str = "LLM"  # Default to LLM; options: 'LLM', 'VLM'

    # LLM Settings
    LLM_MODEL_NAME: str = "mistralai/Mistral-Small-Instruct-2409"
    IS_MISTRAL: bool = True

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

    # Embedding Settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    EMBEDDING_DIM: int = 384

    # PostgreSQL/Timescale Settings
    TIMESCALE_SERVICE_URL: str = "postgres://postgres:password@localhost:5432/postgres"
    VECTOR_STORE_TABLE_NAME: str = "embeddings"
    VECTOR_STORE_EMBEDDING_DIMENSIONS: int = 384
    VECTOR_STORE_TIME_PARTITION_INTERVAL: timedelta = timedelta(days=7)

    # RAG
    CHUNK_SIZE: int = 1000

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


settings = Settings()
