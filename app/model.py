import logging
import os
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_community.llms import VLLM as LangChainVLLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, BaseMessageChunk
from langchain_core.outputs import ChatGenerationChunk, GenerationChunk
from pydantic import Field
from vllm import LLM as VLM

from app.config import settings
from app.utils import load_image

logger = logging.getLogger(__name__)

llm = None
vlm = None
image = None


class MockLLM(BaseLanguageModel):
    """Mock LLM for testing purposes"""

    model_name: str = Field(default="mock_llm")
    mock_response_type: str = Field(default="mock")

    def invoke(self, prompt: str, **kwargs) -> dict:
        """Basic mock response that just needs to work with LangChain."""
        return {"response": "SearchObjective"}

    @property
    def _llm_type(self) -> str:
        return "mock_llm"

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        return {"mock": True}

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        return self.invoke(prompt)

    def predict(self, text: str, **kwargs) -> str:
        return self.invoke(text)

    def predict_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        return self.invoke(str(messages))

    async def apredict(self, text: str, **kwargs) -> str:
        return self.invoke(text)

    async def apredict_messages(self, messages: List[BaseMessage], **kwargs) -> str:
        return self.invoke(str(messages))

    def stream(self, prompt: str, **kwargs) -> Iterator[GenerationChunk]:
        yield GenerationChunk(text=self.invoke(prompt))

    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[GenerationChunk]:
        yield GenerationChunk(text=self.invoke(prompt))

    def stream_chat(
        self, messages: List[BaseMessage], **kwargs
    ) -> Iterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=BaseMessageChunk(content=self.invoke(str(messages)))
        )

    async def astream_chat(
        self, messages: List[BaseMessage], **kwargs
    ) -> AsyncIterator[ChatGenerationChunk]:
        yield ChatGenerationChunk(
            message=BaseMessageChunk(content=self.invoke(str(messages)))
        )

    def generate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Mock implementation of generate_prompt"""
        return [{"generated_text": self.invoke(prompt)} for prompt in prompts]

    async def agenerate_prompt(
        self,
        prompts: List[str],
        stop: Optional[List[str]] = None,
        callbacks: Optional[Any] = None,
        **kwargs: Any,
    ) -> Any:
        """Mock implementation of agenerate_prompt"""
        return [{"generated_text": self.invoke(prompt)} for prompt in prompts]


def initialize_models():
    global llm, vlm, image

    if os.getenv("USE_MOCK_LLM", "false").lower() == "true":
        llm = MockLLM()
        logger.info("Initialized Mock LLM for testing")
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
