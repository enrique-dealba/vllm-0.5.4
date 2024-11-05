import logging

from langchain_community.llms import VLLM as LangChainVLLM

logger = logging.getLogger(__name__)


class ModelManager:
    def __init__(self):
        self.llm = None
        self.initialize_llm()

    def initialize_llm(self) -> bool:
        """Initialize the LLM."""
        try:
            from app.config import settings

            cuda_device = settings.CUDA_DEVICE
            logger.info(f"Initializing LLM for device {cuda_device}")

            tokenizer_mode = "mistral" if settings.IS_MISTRAL else "auto"
            device = f"cuda:{cuda_device}"
            logger.info(f"Using device: {device}")

            self.llm = LangChainVLLM(
                model=settings.LLM_MODEL_NAME,
                trust_remote_code=True,
                max_new_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                tokenizer_mode=tokenizer_mode,
                tensor_parallel_size=1,
                vllm_kwargs={
                    "tokenizer_mode": tokenizer_mode,
                    "device": device,
                },
            )

            # Verify LLM is working
            test_output = self.llm.invoke("test")
            if not isinstance(test_output, str):
                raise RuntimeError("LLM test inference failed")

            logger.info(f"Successfully initialized LLM on device {cuda_device}")
            return True

        except Exception as e:
            logger.error(f"Failed to initialize LLM: {e}")
            self.llm = None
            return False

    def get_llm(self):
        return self.llm
