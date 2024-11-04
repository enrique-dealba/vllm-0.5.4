import logging
import os
import sys

import torch
import uvicorn

# Set CUDA device before any other imports
cuda_device = os.environ.get("CUDA_DEVICE", "0")
os.environ["CUDA_VISIBLE_DEVICES"] = cuda_device
print(f"Setting CUDA_VISIBLE_DEVICES={cuda_device}")

logger = logging.getLogger(__name__)

from app.langchain_server import app, verify_gpu_setup


def verify_cuda_setup():
    cuda_device = os.environ.get("CUDA_DEVICE", "0")
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")

    logger.info(f"CUDA_DEVICE={cuda_device}")
    logger.info(f"CUDA_VISIBLE_DEVICES={cuda_visible}")
    logger.info(f"CUDA available: {torch.cuda.is_available()}")
    logger.info(f"CUDA device count: {torch.cuda.device_count()}")

    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(int(cuda_device))
        logger.info(f"GPU {cuda_device} properties: {device_props}")


if __name__ == "__main__":
    # Verify GPU setup before starting
    if not verify_gpu_setup():
        print("GPU setup verification failed!")
        sys.exit(1)

    port = int(os.environ.get("PORT", 8888))
    uvicorn.run(app, host="0.0.0.0", port=port, workers=1)
