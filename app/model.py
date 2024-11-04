import logging
import os
from typing import Optional

import torch
from langchain_community.llms import VLLM as LangChainVLLM

logger = logging.getLogger(__name__)


class ModelManager:
    _instance: Optional["ModelManager"] = None
    llm = None

    @classmethod
    def get_instance(cls) -> "ModelManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def verify_gpu_setup(self) -> bool:
        """Verify GPU setup is correct."""
        try:
            cuda_device = int(os.environ.get("CUDA_DEVICE", "0"))

            if not torch.cuda.is_available():
                logger.error("CUDA not available")
                return False

            if cuda_device >= torch.cuda.device_count():
                logger.error(
                    f"Requested CUDA device {cuda_device} but only {torch.cuda.device_count()} devices available"
                )
                return False

            # Try to allocate a small tensor on the specified device
            try:
                with torch.cuda.device(cuda_device):
                    test_tensor = torch.zeros(1, device=f"cuda:{cuda_device}")
                    del test_tensor
            except Exception as e:
                logger.error(f"Failed to allocate tensor on device {cuda_device}: {e}")
                return False

            logger.info(f"Successfully verified GPU setup for device {cuda_device}")
            return True

        except Exception as e:
            logger.error(f"GPU verification failed: {e}")
            return False

    def initialize_llm(self) -> bool:
        """Initialize the LLM."""
        try:
            from app.config import settings

            cuda_device = int(os.environ.get("CUDA_DEVICE", "0"))
            logger.info(f"Initializing LLM for device {cuda_device}")
            logger.info(
                f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}"
            )
            logger.info(f"Current CUDA device count: {torch.cuda.device_count()}")

            if not self.verify_gpu_setup():
                raise RuntimeError("GPU setup verification failed")

            tokenizer_mode = "mistral" if settings.IS_MISTRAL else "auto"
            self.llm = LangChainVLLM(
                model=settings.LLM_MODEL_NAME,
                trust_remote_code=True,
                max_new_tokens=settings.MAX_TOKENS,
                temperature=settings.TEMPERATURE,
                tokenizer_mode=tokenizer_mode,
                tensor_parallel_size=1,
                vllm_kwargs={
                    "tokenizer_mode": tokenizer_mode,
                    # "gpu_memory_utilization": gpu_utilization,
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


# Initialize the singleton
model_manager = ModelManager.get_instance()
