import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.langchain_structured_outputs import generate_objective_response
from app.llm_logic import generate_response
from app.utils import get_displayable_fields

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


@app.post("/generate_full_objective")
async def generate_full_objective_api(request: Request):
    query = "Unknown query - failed to parse request"  # Default for logging if request parsing fails
    try:
        request_data = await request.json()
        query = request_data.get("text")  # Get query for logging
        if not query:
            logger.warning("/generate_full_objective called with no text in request.")
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        if not settings.USE_STRUCTURED_OUTPUT:
            logger.warning(
                f"/generate_full_objective called but USE_STRUCTURED_OUTPUT is false. Query: '{query[:100]}...'"
            )
            raise HTTPException(
                status_code=400,
                detail="USE_STRUCTURED_OUTPUT must be enabled for objective schema generation.",
            )

        logger.info(f"/generate_full_objective request for query: '{query[:100]}...'")
        # llm_response_data is the actual content (Pydantic model or error string)
        # execution_time is from the @time_function decorator on generate_objective_response
        llm_response_data, execution_time = await run_in_threadpool(
            generate_objective_response, query
        )

        logger.info(
            f"generate_objective_response returned type: {type(llm_response_data)} for query: '{query[:100]}...'"
        )

        if isinstance(
            llm_response_data, str
        ):  # Indicates an error message string was returned
            logger.error(
                f"Error processed by generate_objective_response for query '{query[:100]}...': {llm_response_data}"
            )
            # If error is from user input or specific objective logic, could be 400 or 422.
            # If it's an unexpected internal server error, 500.
            raise HTTPException(status_code=500, detail=llm_response_data)

        # If not a string, it should be the Pydantic model object
        logger.info(
            f"Successfully generated objective for query: '{query[:100]}...'. Preparing displayable fields."
        )
        response_dict = get_displayable_fields(llm_response_data)  # from app.utils
        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        # Re-raise HTTPException so FastAPI handles it and returns the correct status code
        raise he
    except Exception as e:
        # Catch any other unexpected errors (e.g., if run_in_threadpool itself fails, or JSON parsing)
        logger.exception(
            f"Critical unexpected error in /generate_full_objective for query '{query[:100]}...': {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Unexpected server error: {str(e)}"
        )


@app.post("/generate_objective")
async def generate_objective_api(request: Request):
    """Generate a spaceplan objective name using the initialized LLM."""
    try:
        request_data = await request.json()
        query = request_data.get("text")

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        # Ensure structured output is enabled
        if not settings.USE_STRUCTURED_OUTPUT:
            raise HTTPException(
                status_code=400,
                detail="USE_STRUCTURED_OUTPUT must be enabled for objective schema generation.",
            )

        llm_response, execution_time = generate_response(query)

        # Convert response to JSON-serializable format
        response_dict = llm_response.model_dump()
        response_dict["execution_time_seconds"] = round(execution_time, 4)

        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error during objective generation: {e}")
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


@app.get("/test-mock")
async def test_mock():
    try:
        from app.model import llm

        test_response = llm.invoke("test query")
        return JSONResponse(
            {
                "status": "success",
                "llm_type": llm._llm_type,
                "test_response": test_response,
            }
        )
    except Exception as e:
        logger.exception("Error testing mock LLM")
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/settings")
async def get_settings():
    """Return the current application settings"""
    settings_dict = settings.model_dump()
    return JSONResponse(content=settings_dict)
