import logging
import os
from typing import Dict

import torch
from langchain_community.llms import VLLM as LangChainVLLM

logger = logging.getLogger(__name__)


class ModelManager:
    _instances: Dict[int, "ModelManager"] = {}

    @classmethod
    def get_instance(cls, gpu_id: int) -> "ModelManager":
        if gpu_id not in cls._instances:
            cls._instances[gpu_id] = cls(gpu_id)
        return cls._instances[gpu_id]

    def __init__(self, gpu_id: int):
        self.gpu_id = gpu_id
        self.llm = None
        self._setup_gpu_environment(gpu_id)
        self.initialize_llm()

    def _setup_gpu_environment(self, cuda_device: int) -> None:
        """Strict GPU isolation setup."""
        try:
            # Don't set CUDA_VISIBLE_DEVICES here anymore
            # Let Docker handle GPU visibility
            n_gpus = torch.cuda.device_count()
            logger.info(f"Available GPUs in container: {n_gpus}")

            if cuda_device >= n_gpus:
                raise RuntimeError(
                    f"GPU {cuda_device} requested but only {n_gpus} GPUs available"
                )

            # Just set the device for this instance
            torch.cuda.set_device(cuda_device)

            # Log device properties
            device_props = torch.cuda.get_device_properties(cuda_device)
            logger.info(
                f"Using GPU {cuda_device}: {device_props.name} with {device_props.total_memory/1e9:.2f}GB memory"
            )

            # Initialize device
            with torch.cuda.device(cuda_device):
                torch.zeros(1, device=f"cuda:{cuda_device}")

            logger.info(f"GPU {cuda_device} initialized successfully")

        except Exception as e:
            logger.error(f"GPU setup failed for device {cuda_device}: {e}")
            raise

    def verify_gpu_setup(self) -> bool:
        """Verify GPU setup is correct."""
        try:
            cuda_device = self.gpu_id

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

            cuda_device = self.gpu_id
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
