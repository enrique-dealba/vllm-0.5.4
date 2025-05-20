import asyncio
import logging
import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.langchain_structured_outputs import generate_objective_response
from app.llm_logic import generate_response
from app.utils import get_displayable_fields

# Configure logging
logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain LLM API", version="1.0.0")

# Define a request timeout for LLM operations
LLM_OPERATION_TIMEOUT_SECONDS = 60.0  # <<<<<<<<<<<< Global Timeout Configuration

# --- In-Memory Circuit Breaker State ---
CIRCUIT_MAX_FAILURES = 3
CIRCUIT_RESET_TIMEOUT_SECONDS = 180  # 3 minutes
circuit_is_open = False
circuit_failures_count = 0
circuit_last_failure_time = None


# Authenticate with Hugging Face Hub
if settings.HUGGING_FACE_HUB_TOKEN:
    try:
        login(token=settings.HUGGING_FACE_HUB_TOKEN)
        logger.info("Successfully logged in to HuggingFace Hub.")
    except Exception as e:
        logger.error(f"Failed to authenticate with HuggingFace Hub: {e}")
else:
    logger.warning("HUGGING_FACE_HUB_TOKEN not provided.")


async def check_circuit_breaker(query_info: str):
    """Checks the circuit breaker and raises HTTPException if open."""
    global circuit_is_open, circuit_failures_count, circuit_last_failure_time
    if circuit_is_open:
        if (
            circuit_last_failure_time
            and (time.time() - circuit_last_failure_time)
            > CIRCUIT_RESET_TIMEOUT_SECONDS
        ):
            logger.info(
                f"Circuit breaker reset timeout reached for {query_info}. Moving to half-open state."
            )
            circuit_is_open = False  # Half-open: allow next request
            circuit_failures_count = 0
        else:
            logger.warning(
                f"Circuit breaker is OPEN. Rejecting request for {query_info}"
            )
            raise HTTPException(
                status_code=503,
                detail="LLM service temporarily unavailable (circuit open). Please try again later.",
            )


async def handle_llm_success():
    """Resets circuit breaker on success."""
    global circuit_is_open, circuit_failures_count
    if circuit_failures_count > 0:
        logger.info("LLM request successful, resetting circuit breaker failure count.")
        circuit_failures_count = 0
    if (
        circuit_is_open
    ):  # If it was open and this is the first success in half-open state
        logger.info("LLM request successful, closing circuit breaker.")
        circuit_is_open = False


async def handle_llm_failure(error_type: str):
    """Handles LLM failure for circuit breaker."""
    global circuit_is_open, circuit_failures_count, circuit_last_failure_time
    circuit_failures_count += 1
    circuit_last_failure_time = time.time()
    logger.warning(f"LLM {error_type}. Consecutive failures: {circuit_failures_count}")
    if circuit_failures_count >= CIRCUIT_MAX_FAILURES and not circuit_is_open:
        logger.error(
            f"Circuit breaker OPENED due to {circuit_failures_count} consecutive {error_type}s."
        )
        circuit_is_open = True


@app.post("/generate")
async def generate_response_api(request: Request):
    query = "Unknown query - /generate endpoint"
    query_info_for_cb = "/generate endpoint"  # For circuit breaker logging
    try:
        await check_circuit_breaker(query_info_for_cb)  # Check circuit breaker

        request_data = await request.json()
        query = request_data.get("text")
        query_info_for_cb = f"/generate for query '{query[:30]}...'"

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        try:
            llm_response, execution_time = await asyncio.wait_for(
                run_in_threadpool(generate_response, query),
                timeout=LLM_OPERATION_TIMEOUT_SECONDS,
            )
            await handle_llm_success()  # Mark success for circuit breaker
        except asyncio.TimeoutError:
            logger.error(
                f"LLM operation for {query_info_for_cb} timed out after {LLM_OPERATION_TIMEOUT_SECONDS}s."
            )
            await handle_llm_failure("timeout")
            raise HTTPException(
                status_code=504,
                detail=f"Request to LLM (generic) timed out after {LLM_OPERATION_TIMEOUT_SECONDS} seconds.",
            )
        except (
            Exception
        ) as e:  # Catch errors from generate_response itself before handling type
            logger.error(
                f"Error from generate_response for {query_info_for_cb}: {e}",
                exc_info=True,
            )
            await handle_llm_failure("execution error")
            raise HTTPException(
                status_code=500,
                detail=f"Error during LLM (generic) processing: {str(e)}",
            )

        # Process response
        if settings.USE_STRUCTURED_OUTPUT and hasattr(llm_response, "model_dump"):
            response_dict = llm_response.model_dump()
        elif isinstance(llm_response, str):
            response_dict = {"response": llm_response}
        elif isinstance(llm_response, dict):
            response_dict = llm_response
        else:
            logger.warning(
                f"Unexpected llm_response type in /generate: {type(llm_response)}. Converting to string."
            )
            response_dict = {"response": str(llm_response)}

        # Check if the successfully returned llm_response is actually an error structure from deeper Pydantic parsing
        # (This logic can be tricky if generate_response wraps errors from generate_structured_response)
        if isinstance(
            response_dict.get("response"), dict
        ) and "error" in response_dict.get(
            "response", {}
        ):  # Check if it's like {"response": {"error": "..."}}
            error_detail = response_dict["response"]["error"]
            logger.error(
                f"generate_response returned a dict with an error key for {query_info_for_cb}: {error_detail}"
            )
            await handle_llm_failure("parsing error")  # Count as failure
            raise HTTPException(status_code=500, detail=error_detail)
        elif isinstance(
            response_dict.get("error"), str
        ):  # Check if it's like {"error": "..."}
            error_detail = response_dict["error"]
            logger.error(
                f"generate_response returned a dict with an error key for {query_info_for_cb}: {error_detail}"
            )
            await handle_llm_failure("parsing error")  # Count as failure
            raise HTTPException(status_code=500, detail=error_detail)

        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:  # Catch other unexpected errors like request.json() failure
        logger.exception(
            f"Outer unexpected error during /generate for {query_info_for_cb}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_full_objective")
async def generate_full_objective_api(request: Request):
    query = "Unknown query - /generate_full_objective endpoint"
    query_info_for_cb = "/generate_full_objective endpoint"
    try:
        await check_circuit_breaker(query_info_for_cb)  # Check circuit breaker

        request_data = await request.json()
        query = request_data.get("text")
        query_info_for_cb = f"/generate_full_objective for query '{query[:30]}...'"

        if not query:
            logger.warning("/generate_full_objective called with no text in request.")
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        if not settings.USE_STRUCTURED_OUTPUT:
            logger.warning(
                f"/generate_full_objective called but USE_STRUCTURED_OUTPUT is false for {query_info_for_cb}"
            )
            raise HTTPException(
                status_code=400, detail="USE_STRUCTURED_OUTPUT must be enabled."
            )

        logger.info(f"Processing {query_info_for_cb}")

        try:
            llm_response_data, execution_time = await asyncio.wait_for(
                run_in_threadpool(generate_objective_response, query),
                timeout=LLM_OPERATION_TIMEOUT_SECONDS,
            )
            await handle_llm_success()  # Mark success for circuit breaker
        except asyncio.TimeoutError:
            logger.error(
                f"LLM operation for {query_info_for_cb} timed out after {LLM_OPERATION_TIMEOUT_SECONDS}s."
            )
            await handle_llm_failure("timeout")
            raise HTTPException(
                status_code=504,
                detail=f"Request to LLM (objective) timed out after {LLM_OPERATION_TIMEOUT_SECONDS} seconds.",
            )
        except Exception as e:  # Catch errors from generate_objective_response itself
            logger.error(
                f"Error from generate_objective_response for {query_info_for_cb}: {e}",
                exc_info=True,
            )
            await handle_llm_failure("execution error")  # Count as failure
            raise HTTPException(
                status_code=500,
                detail=f"Error during LLM (objective) processing: {str(e)}",
            )

        logger.info(
            f"generate_objective_response returned type: {type(llm_response_data)} for {query_info_for_cb}"
        )

        if isinstance(llm_response_data, str):
            logger.error(
                f"Error string returned by generate_objective_response for {query_info_for_cb}: {llm_response_data}"
            )
            # This error string *is* the failure, circuit breaker already handled if it was timeout/exec error
            # If it's a "graceful" error string from logic, it might not be a "failure" for CB if not timeout.
            # However, consistent error strings are still service failures.
            # Let's assume any string response here IS a failure for circuit breaker counting too
            # if not already counted by TimeoutError or execution Exception.
            # The problem is, this part is reached *after* potential handle_llm_success if no direct exception.
            # This needs careful thought: is a "logical error string" a "failure" for CB? For now, let's say yes.
            # To do this cleanly, CB handling should be more integrated with outcome.
            # For now, the explicit raise HTTPException is the main error path.
            raise HTTPException(status_code=500, detail=llm_response_data)

        logger.info(
            f"Successfully generated objective for {query_info_for_cb}. Preparing displayable fields."
        )
        response_dict = get_displayable_fields(llm_response_data)
        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(
            f"Outer unexpected error in /generate_full_objective for {query_info_for_cb}: {e}"
        )
        raise HTTPException(
            status_code=500, detail=f"Unexpected server error: {str(e)}"
        )


@app.post("/generate_objective")
async def generate_objective_api(request: Request):
    query = "Unknown query - /generate_objective endpoint"
    query_info_for_cb = "/generate_objective endpoint"
    try:
        await check_circuit_breaker(query_info_for_cb)

        request_data = await request.json()
        query = request_data.get("text")
        query_info_for_cb = f"/generate_objective for query '{query[:30]}...'"

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        if not settings.USE_STRUCTURED_OUTPUT:
            raise HTTPException(
                status_code=400, detail="USE_STRUCTURED_OUTPUT must be enabled."
            )

        try:
            llm_response, execution_time = await asyncio.wait_for(
                run_in_threadpool(generate_response, query),
                timeout=LLM_OPERATION_TIMEOUT_SECONDS,
            )
            await handle_llm_success()
        except asyncio.TimeoutError:
            logger.error(
                f"LLM operation for {query_info_for_cb} timed out after {LLM_OPERATION_TIMEOUT_SECONDS}s."
            )
            await handle_llm_failure("timeout")
            raise HTTPException(
                status_code=504,
                detail=f"Request to LLM (objective name) timed out after {LLM_OPERATION_TIMEOUT_SECONDS} seconds.",
            )
        except Exception as e:  # Catch errors from generate_response itself
            logger.error(
                f"Error from generate_response for {query_info_for_cb}: {e}",
                exc_info=True,
            )
            await handle_llm_failure("execution error")
            raise HTTPException(
                status_code=500,
                detail=f"Error during LLM (objective name) processing: {str(e)}",
            )

        # Process response
        if settings.USE_STRUCTURED_OUTPUT and hasattr(llm_response, "model_dump"):
            response_dict = llm_response.model_dump()
        elif isinstance(llm_response, dict):
            response_dict = llm_response
        else:
            logger.error(
                f"Expected Pydantic model or dict for /generate_objective, got {type(llm_response)} for {query_info_for_cb}"
            )
            # This case might indicate an error string was returned by generate_response without an exception
            if isinstance(llm_response, str):
                await handle_llm_failure("unexpected string error")
                raise HTTPException(
                    status_code=500, detail=llm_response
                )  # Treat string as error
            raise HTTPException(
                status_code=500,
                detail="Internal server error: Unexpected response type from LLM.",
            )

        # Check if the successfully returned llm_response is actually an error structure
        if isinstance(
            response_dict.get("error"), str
        ):  # Check if it's like {"error": "..."}
            error_detail = response_dict["error"]
            logger.error(
                f"generate_response (for /generate_objective) returned a dict with an error key for {query_info_for_cb}: {error_detail}"
            )
            # await handle_llm_failure("parsing error") # Already counted if exception, this is for logical error returns
            raise HTTPException(status_code=500, detail=error_detail)

        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(
            f"Outer unexpected error during /generate_objective for {query_info_for_cb}: {e}"
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of the model service."""
    from app.model import llm, vlm

    # Check circuit breaker status as part of health
    # If circuit is open, service is effectively unhealthy from application perspective
    global circuit_is_open, circuit_last_failure_time, CIRCUIT_RESET_TIMEOUT_SECONDS
    if circuit_is_open:
        # Check if it's time to try resetting (half-open)
        if (
            circuit_last_failure_time
            and (time.time() - circuit_last_failure_time)
            > CIRCUIT_RESET_TIMEOUT_SECONDS
        ):
            logger.info(
                "Health check: Circuit breaker is in reset period (half-open). Allowing potential recovery."
            )
            # Service might be considered "degraded" but attempting recovery.
            # For simplicity, let's still report as unhealthy until a request succeeds.
            # Or, you could have a specific "degraded" status.
        else:
            logger.warning("Health check: Circuit breaker is OPEN.")
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "LLM service circuit breaker is open due to repeated failures.",
                },
            )

    if settings.MODEL_TYPE.upper() == "LLM":
        if llm is None:
            logger.warning("LLM is not initialized for health check.")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "LLM is not initialized."},
            )

        if hasattr(llm, "ainvoke"):
            try:
                test_prompt = "Briefly confirm status with 'OK'."
                logger.debug("Attempting async LLM inference for health check...")
                response = await asyncio.wait_for(
                    llm.ainvoke(test_prompt, max_new_tokens=5), timeout=15.0
                )  # Generous timeout for health check
                if (
                    "ok" not in str(response).lower()
                ):  # Check if 'OK' is in the response
                    logger.warning(
                        f"LLM health check: Unexpected response from ainvoke: '{str(response)[:100]}'"
                    )
                    # Not necessarily unhealthy if it responds, but not the expected probe response
                logger.info("Async LLM inference health check probe attempt finished.")
            except asyncio.TimeoutError:
                logger.error("LLM health check: async inference timed out.")
                # This is a strong signal of unhealthiness.
                await handle_llm_failure(
                    "health check timeout"
                )  # Trip circuit breaker on health check timeout
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "message": "LLM inference timed out during health check.",
                    },
                )
            except Exception as e:
                logger.error(
                    f"LLM health check: async inference failed: {str(e)}", exc_info=True
                )
                await handle_llm_failure(
                    "health check error"
                )  # Trip circuit breaker on health check error
                return JSONResponse(
                    status_code=503,
                    content={
                        "status": "unhealthy",
                        "message": f"LLM inference failed during health check: {str(e)}.",
                    },
                )
        else:
            logger.info(
                "LLM is initialized (sync model, advanced health inference test via ainvoke not available)."
            )

    elif settings.MODEL_TYPE.upper() == "VLM":
        # Similar logic for VLM
        if vlm is None:
            logger.warning("VLM is not initialized.")
            return JSONResponse(
                status_code=503,
                content={"status": "unhealthy", "message": "VLM is not initialized."},
            )
        logger.info("VLM is initialized (basic check).")
    else:
        logger.error(f"Invalid MODEL_TYPE '{settings.MODEL_TYPE}' for health check.")
        return JSONResponse(
            status_code=500,
            content={"status": "invalid", "message": "Invalid MODEL_TYPE."},
        )

    # If we reached here and circuit isn't open, and basic checks passed.
    # If an ainvoke check passed or wasn't performed for sync, consider it healthy.
    await (
        handle_llm_success()
    )  # If health check involved successful LLM call, reset CB failures
    logger.info("Model service health check passed.")
    return JSONResponse(
        {"status": "healthy", "message": "Model is initialized and ready."}
    )


@app.get("/test-mock")
async def test_mock():
    # ... (existing code) ...
    try:
        from app.model import llm

        if llm._llm_type != "mock_llm":  # Ensure we only call invoke on actual mock
            raise HTTPException(
                status_code=400, detail="This endpoint is for mock LLM only."
            )

        test_response = llm.invoke("test query")
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
async def get_settings_api():
    """Return the current application settings"""
    settings_dict = settings.model_dump()
    return JSONResponse(content=settings_dict)
