from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    MODEL_NAME: str = "llava-hf/llava-1.5-7b-hf"
    FIXED_IMAGE_URL: str = "https://upload.wikimedia.org/wikipedia/commons/thumb/d/dd/Gfp-wisconsin-madison-the-nature-boardwalk.jpg/2560px-Gfp-wisconsin-madison-the-nature-boardwalk.jpg"
    IMAGE_FETCH_TIMEOUT: int = 5
    PORT: int = 8888
    HOST: str = "0.0.0.0"
    TEMPERATURE: float = 0.2
    MAX_TOKENS: int = 256

    class Config:
        env_file = ".env"


settings = Settings()