import logging
import os
import sys
import traceback
from pathlib import Path

import torch

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    logger.info(f"Python path: {sys.path}")
    logger.info(f"Current directory: {os.getcwd()}")
    from app.langchain_server import app
except Exception as e:
    logger.error(f"Import error: {str(e)}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    sys.exit(1)


def check_gpu_availability():
    try:
        if not torch.cuda.is_available():
            logger.error("CUDA is not available")
            return False
        gpu_count = torch.cuda.device_count()
        logger.info(f"Found {gpu_count} GPU(s)")
        for i in range(gpu_count):
            logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        return True
    except Exception as e:
        logger.error(f"Error checking GPU availability: {e}")
        return False


def verify_environment():
    required_vars = ["HUGGING_FACE_HUB_TOKEN", "LANGCHAIN_API_KEY"]
    missing = [var for var in required_vars if not os.getenv(var)]
    if missing:
        logger.error(f"Missing required environment variables: {missing}")
        return False
    logger.info("Environment variables verified")
    return True


def verify_gpu_setup():
    """Verify GPU setup before starting server."""
    try:
        n_gpus = torch.cuda.device_count()
        logger.info(f"Available GPUs: {n_gpus}")

        if n_gpus < 1:
            logger.error(f"Need at least 1 GPU, found {n_gpus}")
            return False

        for i in range(n_gpus):
            props = torch.cuda.get_device_properties(i)
            logger.info(f"GPU {i}: {props.name} ({props.total_memory/1e9:.2f}GB)")

        return True
    except Exception as e:
        logger.error(f"GPU verification failed: {e}")
        return False


def set_cuda_visible_devices():
    from app.config import settings

    os.environ["CUDA_VISIBLE_DEVICES"] = str(settings.CUDA_DEVICE)
    logger.info(f"Set CUDA_VISIBLE_DEVICES to {os.environ['CUDA_VISIBLE_DEVICES']}")


if __name__ == "__main__":
    try:
        logger.info("Starting multi-LLM service initialization...")

        # Check GPUs first
        if not verify_gpu_setup():
            sys.exit(1)

        # Check environment
        if not verify_environment():
            sys.exit(1)

        # Check GPU
        if not check_gpu_availability():
            sys.exit(1)

        set_cuda_visible_devices()

        # Get port
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
        logger.info(f"Starting server on port {port}")

        # Start server with proper config
        import uvicorn

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
            timeout_keep_alive=65,
            limit_concurrency=100,
            limit_max_requests=100000,
            backlog=2048,
            timeout_graceful_shutdown=30,
        )
        server = uvicorn.Server(config)
        server.run()
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
