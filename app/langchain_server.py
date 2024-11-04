import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login

from app.config import settings
from app.llm_logic import generate_response

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain LLM API", version="1.0.0")

# Authenticate with Hugging Face Hub
if settings.HUGGING_FACE_HUB_TOKEN:
    try:
        login(token=settings.HUGGING_FACE_HUB_TOKEN)
        logger.info("Successfully logged in to HuggingFace Hub.")
    except Exception as e:
        logger.error(f"Failed to authenticate with HuggingFace Hub: {e}")
else:
    logger.warning("HUGGING_FACE_HUB_TOKEN not provided.")


@app.post("/generate")
async def generate_response_api(request: Request):
    """Generate a response using the initialized LLM with optional peer call."""
    try:
        request_data = await request.json()
        query = request_data.get("text")
        call_peer = request_data.get("call_peer", False)

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        llm_response, execution_time = await generate_response(
            query, call_peer=call_peer
        )

        if settings.USE_STRUCTURED_OUTPUT:
            response_dict = llm_response.model_dump()
        else:
            response_dict = {"response": llm_response}

        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of the model service."""
    from app.model import check_gpu_status, llm, vlm

    try:
        # Check GPU status
        check_gpu_status()

        if settings.MODEL_TYPE.upper() == "LLM":
            if llm is None:
                logger.warning(
                    f"LLM is not initialized for service {settings.SERVICE_NAME}"
                )
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "message": f"LLM not initialized for service {settings.SERVICE_NAME}",
                        "service_name": settings.SERVICE_NAME,
                        "cuda_device": settings.CUDA_DEVICE,
                        "model_type": settings.MODEL_TYPE,
                    },
                )

            # Try a simple inference to verify LLM is working
            try:
                _ = llm.invoke("test")
                logger.info(
                    f"LLM health check passed for service {settings.SERVICE_NAME}"
                )
            except Exception as e:
                logger.error(f"LLM inference test failed: {str(e)}")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "message": f"LLM inference test failed: {str(e)}",
                        "service_name": settings.SERVICE_NAME,
                        "cuda_device": settings.CUDA_DEVICE,
                        "model_type": settings.MODEL_TYPE,
                    },
                )

        elif settings.MODEL_TYPE.upper() == "VLM":
            if vlm is None:
                logger.warning("VLM is not initialized")
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "message": "VLM is not initialized. GPU may not be available.",
                        "service_name": settings.SERVICE_NAME,
                        "cuda_device": settings.CUDA_DEVICE,
                        "model_type": settings.MODEL_TYPE,
                    },
                )

        else:
            logger.error(f"Invalid MODEL_TYPE: {settings.MODEL_TYPE}")
            return JSONResponse(
                status_code=500,
                content={
                    "status": "invalid",
                    "message": f"Invalid MODEL_TYPE: {settings.MODEL_TYPE}",
                    "service_name": settings.SERVICE_NAME,
                    "model_type": settings.MODEL_TYPE,
                },
            )

        # If we get here, service is healthy
        logger.info(f"Model service {settings.SERVICE_NAME} is healthy")
        return JSONResponse(
            {
                "status": "healthy",
                "message": f"Model is initialized and ready on CUDA device {settings.CUDA_DEVICE}",
                "service_name": settings.SERVICE_NAME,
                "model_type": settings.MODEL_TYPE,
                "cuda_device": settings.CUDA_DEVICE,
            }
        )

    except Exception as e:
        logger.error(f"Health check failed with error: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={
                "status": "error",
                "message": str(e),
                "service_name": settings.SERVICE_NAME,
                "model_type": settings.MODEL_TYPE,
                "cuda_device": settings.CUDA_DEVICE,
            },
        )
