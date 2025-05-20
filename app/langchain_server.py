import asyncio
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

LLM_OPERATION_TIMEOUT_SECONDS = 15.0

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
    query = "Unknown query - /generate endpoint"
    try:
        request_data = await request.json()
        query = request_data.get("text")

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        try:
            # generate_response is from llm_logic.py
            llm_response, execution_time = await asyncio.wait_for(
                run_in_threadpool(generate_response, query),
                timeout=LLM_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:
            logger.error(
                f"LLM operation for /generate query '{query[:100]}...' timed out after {LLM_OPERATION_TIMEOUT_SECONDS}s."
            )
            raise HTTPException(
                status_code=504,
                detail=f"Request to LLM (generic) timed out after {LLM_OPERATION_TIMEOUT_SECONDS} seconds. The model may be overloaded or stuck.",
            )

        if settings.USE_STRUCTURED_OUTPUT and hasattr(llm_response, "model_dump"):
            response_dict = llm_response.model_dump()
        elif isinstance(llm_response, str):
            response_dict = {"response": llm_response}
        elif isinstance(
            llm_response, dict
        ):  # If generate_response already returned a dict
            response_dict = llm_response
            # Check if this dict is an error structure from a deeper layer
            if "error" in response_dict:
                logger.error(
                    f"Error dict returned by generate_response for /generate query '{query[:100]}...': {response_dict['error']}"
                )
                raise HTTPException(status_code=500, detail=str(response_dict["error"]))
        else:
            logger.warning(
                f"Unexpected llm_response type in /generate: {type(llm_response)}. Converting to string."
            )
            response_dict = {"response": str(llm_response)}

        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(
            f"Outer unexpected error during /generate for query '{query}': {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_full_objective")
async def generate_full_objective_api(request: Request):
    query = "Unknown query - /generate_full_objective endpoint"
    try:
        request_data = await request.json()
        query = request_data.get("text")
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

        try:
            (
                llm_response_data,
                execution_time,
            ) = await asyncio.wait_for(  # <<<<<<<<<<<< ADDED WRAPPER
                run_in_threadpool(generate_objective_response, query),
                timeout=LLM_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:  # <<<<<<<<<<<< ADDED TIMEOUT HANDLING
            logger.error(
                f"LLM operation for /generate_full_objective query '{query[:100]}...' timed out after {LLM_OPERATION_TIMEOUT_SECONDS}s."
            )
            raise HTTPException(
                status_code=504,
                detail=f"Request to LLM (objective) timed out after {LLM_OPERATION_TIMEOUT_SECONDS} seconds. The model may be overloaded or stuck.",
            )

        logger.info(
            f"generate_objective_response returned type: {type(llm_response_data)} for query: '{query[:100]}...'"
        )

        if isinstance(llm_response_data, str):
            logger.error(
                f"Error string returned by generate_objective_response for query '{query[:100]}...': {llm_response_data}"
            )
            raise HTTPException(status_code=500, detail=llm_response_data)

        logger.info(
            f"Successfully generated objective for query: '{query[:100]}...'. Preparing displayable fields."
        )
        response_dict = get_displayable_fields(llm_response_data)
        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(
            f"Critical unexpected error in /generate_full_objective for query '{query[:100]}...': {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Unexpected server error: {str(e)}"
        )


@app.post("/generate_objective")
async def generate_objective_api(request: Request):
    """Generate a spaceplan objective name using the initialized LLM."""
    query = "Unknown query - /generate_objective endpoint"
    try:
        request_data = await request.json()
        query = request_data.get("text")

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        if not settings.USE_STRUCTURED_OUTPUT:
            raise HTTPException(
                status_code=400,
                detail="USE_STRUCTURED_OUTPUT must be enabled for objective schema generation.",
            )

        try:
            # generate_response is from llm_logic.py
            (
                llm_response,
                execution_time,
            ) = await asyncio.wait_for(  # <<<<<<<<<<<< ADDED WRAPPER
                run_in_threadpool(generate_response, query),
                timeout=LLM_OPERATION_TIMEOUT_SECONDS,
            )
        except asyncio.TimeoutError:  # <<<<<<<<<<<< ADDED TIMEOUT HANDLING
            logger.error(
                f"LLM operation for /generate_objective query '{query[:100]}...' timed out after {LLM_OPERATION_TIMEOUT_SECONDS}s."
            )
            raise HTTPException(
                status_code=504,
                detail=f"Request to LLM (objective name) timed out after {LLM_OPERATION_TIMEOUT_SECONDS} seconds.",
            )

        # Ensure llm_response is Pydantic before model_dump, or handle other types
        if settings.USE_STRUCTURED_OUTPUT and hasattr(llm_response, "model_dump"):
            response_dict = llm_response.model_dump()
        elif isinstance(llm_response, str):  # For non-structured output
            response_dict = {"response": llm_response}
        elif isinstance(
            llm_response, dict
        ):  # If generate_response itself returns a dict (e.g. error)
            response_dict = llm_response
            # Check if this dict is an error structure from a deeper layer
            if "error" in response_dict:
                logger.error(
                    f"Error dict returned by generate_response for /generate_objective query '{query[:100]}...': {response_dict['error']}"
                )
                raise HTTPException(status_code=500, detail=str(response_dict["error"]))
        else:
            logger.error(
                f"Expected Pydantic model or dict for /generate_objective with USE_STRUCTURED_OUTPUT=True, got {type(llm_response)}"
            )
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Unexpected response type from LLM for objective name.",
            )

        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(
            f"Outer unexpected error during /generate_objective for query '{query}': {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of the model service."""
    from app.model import llm, vlm  # Ensure llm is imported

    # Basic health check: service is up and model object is initialized.
    # Avoids complex inference that might hang if LLM is in a sensitive state.
    if settings.MODEL_TYPE.upper() == "LLM":
        if llm is None:
            logger.warning("LLM is not initialized for health check.")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "LLM is not initialized. GPU may not be available.",
                },
            )
    elif settings.MODEL_TYPE.upper() == "VLM":
        if vlm is None:  # Assuming 'vlm' is your VLM instance
            logger.warning("VLM is not initialized for health check.")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "VLM is not initialized. GPU may not be available.",
                },
            )
    else:
        logger.error(
            f"Invalid MODEL_TYPE '{settings.MODEL_TYPE}' configuration for health check."
        )
        return JSONResponse(
            status_code=500,
            content={
                "status": "invalid",
                "message": "Invalid MODEL_TYPE configuration.",
            },
        )

    logger.info("Model service health check passed (basic initialization check).")
    return JSONResponse(
        {
            "status": "healthy",
            "message": "Model is initialized and ready (basic check).",
        }
    )


@app.get("/test-mock")
async def test_mock():
    try:
        from app.model import llm

        # Ensure it's actually the mock LLM to prevent calling invoke on a real model here
        if not hasattr(llm, "_llm_type") or llm._llm_type != "mock_llm":
            raise HTTPException(
                status_code=400, detail="This endpoint is for mock LLM only."
            )

        test_response = llm.invoke(
            "test query"
        )  # Mock LLM's invoke should be safe & fast
        return JSONResponse(
            {
                "status": "success",
                "llm_type": llm._llm_type,
                "test_response": test_response,
            }
        )
    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception("Error testing mock LLM")
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/settings")
async def get_settings():
    """Return the current application settings"""
    settings_dict = settings.model_dump()
    return JSONResponse(content=settings_dict)
