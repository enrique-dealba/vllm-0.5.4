import logging
import os
import sys
import traceback

# Setup logging
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

from app.langchain_server import app


def check_gpu_availability():
    try:
        import torch

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


if __name__ == "__main__":
    try:
        logger.info("Starting multi-LLM service initialization...")

        # Check environment
        if not verify_environment():
            sys.exit(1)

        # Check GPU
        if not check_gpu_availability():
            sys.exit(1)

        # Get port
        port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
        logger.info(f"Starting server on port {port}")

        # Start server
        import uvicorn

        uvicorn.run(
            app,
            host="0.0.0.0",
            port=port,
            workers=1,
            log_level=os.getenv("LOG_LEVEL", "info").lower(),
        )
    except Exception as e:
        logger.error(f"Failed to start server: {str(e)}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        sys.exit(1)
