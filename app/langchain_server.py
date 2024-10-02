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
    """Generate a response using the initialized LLM or VLM."""
    try:
        request_data = await request.json()
        query = request_data.get("text")

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        llm_response, execution_time = generate_response(query)

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
    from app.model import llm, vlm

    if settings.MODEL_TYPE.upper() == "LLM":
        if llm is None:
            logger.warning("LLM is not initialized.")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "LLM is not initialized. GPU may not be available.",
                },
            )
    elif settings.MODEL_TYPE.upper() == "VLM":
        if vlm is None:
            logger.warning("VLM is not initialized.")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "VLM is not initialized. GPU may not be available.",
                },
            )
    else:
        logger.error("Invalid MODEL_TYPE configuration.")
        return JSONResponse(
            status_code=500,
            content={
                "status": "invalid",
                "message": "Invalid MODEL_TYPE configuration.",
            },
        )

    logger.info("Model service is healthy.")
    return JSONResponse(
        {"status": "healthy", "message": "Model is initialized and ready."}
    )
