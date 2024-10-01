import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login
from vllm import SamplingParams

from app.config import settings
from app.model import image, llm, vlm

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
        # llm and vlm are already set to None in model.py
else:
    logger.warning("HUGGING_FACE_HUB_TOKEN not provided.")
    # llm and vlm are already set to None in model.py


@app.post("/generate")
async def generate_response(request: Request):
    """Generate a response using the initialized LLM or VLM."""
    try:
        request_data = await request.json()
        query = request_data.get("text")

        if not query:
            logger.warning("No text provided for generation.")
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        start_time = time.time()

        if settings.MODEL_TYPE.upper() == "LLM":
            if llm is None:
                logger.error("LLM is not available.")
                raise HTTPException(status_code=503, detail="LLM is not available.")

            # Generate text using LangChain's VLLM
            generated_text = llm.invoke(query)

        elif settings.MODEL_TYPE.upper() == "VLM":
            if vlm is None:
                logger.error("VLM is not available.")
                raise HTTPException(status_code=503, detail="VLM is not available.")

            prompt = f"USER: <image>\n{query}\nASSISTANT:"
            sampling_params = SamplingParams(
                temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS
            )
            outputs = vlm.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    }
                ],
                sampling_params,
            )
            generated_text = outputs[0].outputs[0].text

        else:
            logger.error("Invalid MODEL_TYPE configuration.")
            raise HTTPException(
                status_code=500, detail="Invalid MODEL_TYPE configuration."
            )

        end_time = time.time()
        execution_time = end_time - start_time

        logger.info(f"Generated response in {execution_time:.4f} seconds.")

        return JSONResponse(
            {
                "response": generated_text,
                "execution_time_seconds": round(execution_time, 4),
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of the model service."""
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
