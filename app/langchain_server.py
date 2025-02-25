import inspect
import json
import logging
import os
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login
from starlette.concurrency import run_in_threadpool

from app.config import settings
from app.langchain_structured_outputs import (
    generate_objective_response,
    generate_objective_response_with_tracking,
    generate_structured_response_with_tracking,
)
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

        # Offload the blocking call to a thread
        llm_response, execution_time = await run_in_threadpool(
            generate_objective_response, query
        )

        response_dict = get_displayable_fields(llm_response)
        response_dict["execution_time_seconds"] = round(execution_time, 4)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error during full objective generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


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


@app.get("/debug_register_hooks")
def debug_register_hooks():
    from app.interpretability_analysis import register_llama_hooks
    from app.model import llm

    try:
        # Retrieve the underlying model from the VLLM wrapper.
        underlying_model = (
            llm.client.llm_engine.model_executor.driver_worker.model_runner.model
        )
        # Register hooks on the underlying model.
        hook_handles = register_llama_hooks(underlying_model)
        # Optionally, you could run a forward pass here to trigger the hooks.
        return JSONResponse(
            {"status": "success", "num_hooks_registered": len(hook_handles)}
        )
    except Exception as e:
        return JSONResponse({"status": "error", "error": str(e)})


@app.get("/debug_model_runner")
def debug_model_runner():
    from app.model import llm

    # Get the model_executor from the LLM engine.
    executor = getattr(llm.client.llm_engine, "model_executor", None)
    if executor is None:
        return JSONResponse({"error": "No model_executor found"})

    # Get the driver_worker from the executor.
    driver_worker = getattr(executor, "driver_worker", None)
    if driver_worker is None:
        return JSONResponse({"error": "No driver_worker found in model_executor"})

    # Get the model_runner from the driver_worker.
    model_runner = getattr(driver_worker, "model_runner", None)
    if model_runner is None:
        return JSONResponse({"error": "No model_runner found in driver_worker"})

    # Prepare debug info for the model_runner.
    model_runner_attrs = {
        attr: str(getattr(model_runner, attr))
        for attr in dir(model_runner)
        if not attr.startswith("_")
    }

    # Attempt to inspect a potential "model" attribute on model_runner.
    model_info = {}
    if hasattr(model_runner, "model"):
        model = getattr(model_runner, "model")
        model_attrs = {
            attr: str(getattr(model, attr))
            for attr in dir(model)
            if not attr.startswith("_")
        }
        model_info = {"type": str(type(model)), "attributes": model_attrs}

    return JSONResponse(
        {
            "model_runner": {
                "type": str(type(model_runner)),
                "attributes": model_runner_attrs,
                "underlying_model": model_info,
            }
        }
    )


@app.get("/debug_executor")
def debug_executor():
    from app.model import llm

    executor = getattr(llm.client.llm_engine, "model_executor", None)
    if executor is None:
        return JSONResponse({"error": "No model_executor found"})

    executor_attrs = {
        attr: str(getattr(executor, attr))
        for attr in dir(executor)
        if not attr.startswith("_")
    }
    # Optionally, try to further inspect a potential "model" attribute:
    model_info = {}
    if hasattr(executor, "model"):
        model = getattr(executor, "model")
        model_attrs = {
            attr: str(getattr(model, attr))
            for attr in dir(model)
            if not attr.startswith("_")
        }
        model_info = {"type": str(type(model)), "attributes": model_attrs}

    return JSONResponse(
        {
            "executor": {
                "type": str(type(executor)),
                "attributes": executor_attrs,
                "underlying_model": model_info,
            }
        }
    )


@app.get("/debug_llm")
def debug_llm():
    from app.model import llm

    debug_info = {}

    # Top-level VLLM object.
    llm_attrs = {
        attr: str(getattr(llm, attr)) for attr in dir(llm) if not attr.startswith("_")
    }
    llm_methods = [
        attr
        for attr in dir(llm)
        if not attr.startswith("_") and inspect.ismethod(getattr(llm, attr))
    ]
    debug_info["llm"] = {
        "type": str(type(llm)),
        "attributes": llm_attrs,
        "methods": llm_methods,
    }

    # Inspect the client.
    if hasattr(llm, "client"):
        client = getattr(llm, "client")
        client_attrs = {
            attr: str(getattr(client, attr))
            for attr in dir(client)
            if not attr.startswith("_")
        }
        client_methods = [
            attr
            for attr in dir(client)
            if not attr.startswith("_") and inspect.ismethod(getattr(client, attr))
        ]
        debug_info["client"] = {
            "type": str(type(client)),
            "attributes": client_attrs,
            "methods": client_methods,
        }

        # Inspect the llm_engine within the client.
        if hasattr(client, "llm_engine"):
            engine = getattr(client, "llm_engine")
            engine_attrs = {
                attr: str(getattr(engine, attr))
                for attr in dir(engine)
                if not attr.startswith("_")
            }
            engine_methods = [
                attr
                for attr in dir(engine)
                if not attr.startswith("_") and inspect.ismethod(getattr(engine, attr))
            ]
            debug_info["client"]["llm_engine"] = {
                "type": str(type(engine)),
                "attributes": engine_attrs,
                "methods": engine_methods,
            }

            # Inspect the GPUExecutor (model_executor) if available.
            if hasattr(engine, "model_executor"):
                executor = getattr(engine, "model_executor")
                executor_attrs = {
                    attr: str(getattr(executor, attr))
                    for attr in dir(executor)
                    if not attr.startswith("_")
                }
                executor_methods = [
                    attr
                    for attr in dir(executor)
                    if not attr.startswith("_")
                    and inspect.ismethod(getattr(executor, attr))
                ]
                debug_info["client"]["llm_engine"]["model_executor"] = {
                    "type": str(type(executor)),
                    "attributes": executor_attrs,
                    "methods": executor_methods,
                }

                # Now inspect the driver_worker, where the underlying model is likely stored.
                if hasattr(executor, "driver_worker"):
                    worker = getattr(executor, "driver_worker")
                    worker_attrs = {
                        attr: str(getattr(worker, attr))
                        for attr in dir(worker)
                        if not attr.startswith("_")
                    }
                    worker_methods = [
                        attr
                        for attr in dir(worker)
                        if not attr.startswith("_")
                        and inspect.ismethod(getattr(worker, attr))
                    ]
                    debug_info["client"]["llm_engine"]["model_executor"][
                        "driver_worker"
                    ] = {
                        "type": str(type(worker)),
                        "attributes": worker_attrs,
                        "methods": worker_methods,
                    }

                    # If the driver_worker exposes an attribute that is the underlying model, capture it.
                    if hasattr(worker, "model"):
                        underlying_model = getattr(worker, "model")
                        model_attrs = {
                            attr: str(getattr(underlying_model, attr))
                            for attr in dir(underlying_model)
                            if not attr.startswith("_")
                        }
                        model_methods = [
                            attr
                            for attr in dir(underlying_model)
                            if not attr.startswith("_")
                            and inspect.ismethod(getattr(underlying_model, attr))
                        ]
                        debug_info["client"]["llm_engine"]["model_executor"][
                            "driver_worker"
                        ]["underlying_model"] = {
                            "type": str(type(underlying_model)),
                            "attributes": model_attrs,
                            "methods": model_methods,
                        }
                    else:
                        debug_info["client"]["llm_engine"]["model_executor"][
                            "driver_worker"
                        ]["underlying_model"] = "Not found"
                else:
                    debug_info["client"]["llm_engine"]["model_executor"][
                        "driver_worker"
                    ] = "Not available"
            else:
                debug_info["client"]["llm_engine"]["model_executor"] = "Not available"
        else:
            debug_info["client"]["llm_engine"] = "Not available"
    else:
        debug_info["client"] = "Not available"

    return JSONResponse(debug_info)


@app.post("/generate_objective_tracking")
async def generate_objective_api_tracking(request: Request):
    """Generate a spaceplan objective name using the initialized LLM with tracking."""
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

        # Generate the response and tracking data.
        llm_response, tracking_data = generate_structured_response_with_tracking(query)

        # Generate timestamp for filename
        save_dir = "/app/plots"
        current_time = datetime.now()
        timestamp = current_time.strftime("%m%d%Y_%H%M")
        filename = f"vllm_interp_{timestamp}.json"

        with open(os.path.join(save_dir, filename), "w") as f:
            json.dump(tracking_data, f, indent=2)
        logger.info(f"Saved analysis data to {filename}")

        # Convert the response to a JSON-serializable format.
        # (Assumes llm_response is a Pydantic model with a model_dump method.)
        response_dict = llm_response.model_dump()
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error during objective generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_full_objective_tracking")
async def generate_full_objective_api_tracking(request: Request):
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

        # Offload the blocking call to a thread
        info_dict = await run_in_threadpool(
            generate_objective_response_with_tracking, query
        )

        llm_response = info_dict["detailed_response"]

        # Combine both JSON data into a single dictionary
        combined_data = {"part_1": info_dict["part_1"], "part_2": info_dict["part_2"]}

        # Generate timestamp for filename
        save_dir = "/app/plots"
        current_time = datetime.now()
        timestamp = current_time.strftime("%m%d%Y_%H%M")
        filename = f"vllm_full_{timestamp}.json"

        with open(os.path.join(save_dir, filename), "w") as f:
            json.dump(combined_data, f, indent=2)
        logger.info(f"Saved combined analysis data to {filename}")

        response_dict = get_displayable_fields(llm_response)
        return JSONResponse(response_dict)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Unexpected error during full objective generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate_experiment")
async def generate_full_objective_experiment(request: Request):
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

        # Offload the blocking call to a thread
        info_dict = await run_in_threadpool(
            generate_objective_response_with_tracking, query
        )

        # Create a response structure that will always be valid
        response_data = {
            "status": "error" if info_dict.get("error") else "success",
            "part_1": info_dict.get("part_1"),
            "part_2": info_dict.get("part_2"),
        }

        # Add error if present
        if info_dict.get("error"):
            response_data["error"] = info_dict["error"]
            logger.warning(f"Processing error: {info_dict['error']}")

        # Add LLM response if present
        llm_response = info_dict.get("detailed_response")
        if llm_response is not None:
            try:
                response_data["llm_response"] = get_displayable_fields(llm_response)
            except Exception as field_error:
                response_data["status"] = "error"
                response_data["error"] = (
                    f"Error extracting displayable fields: {str(field_error)}"
                )
                logger.error(f"Field extraction error: {field_error}")

        # Save data regardless of success/failure
        try:
            save_dir = "/app/plots"
            os.makedirs(save_dir, exist_ok=True)
            current_time = datetime.now()
            timestamp = current_time.strftime("%m%d%Y_%H%M")
            filename = f"vllm_full_{timestamp}.json"

            with open(os.path.join(save_dir, filename), "w") as f:
                json.dump(response_data, f, indent=2)
            logger.info(f"Saved data to {filename}")
        except Exception as save_error:
            logger.error(f"Error saving data: {save_error}")

        # Return response regardless of errors
        return JSONResponse(response_data)

    except HTTPException as he:
        raise he
    except Exception as e:
        logger.exception(f"Server error: {e}")
        return JSONResponse(
            {"status": "error", "error": f"Server error: {str(e)}"}, status_code=500
        )
