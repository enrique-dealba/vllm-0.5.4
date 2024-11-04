import logging

from fastapi import FastAPI, HTTPException, Request

from app.router import LLMRouter

logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain LLM API", version="1.0.0")

# Initialize the router with service names and GPU IDs
router = LLMRouter({"llm1": 0, "llm2": 1})


@app.post("/{service_name}/generate")
async def generate(service_name: str, request: Request):
    try:
        request_data = await request.json()
        response = await router.forward_request(service_name, request_data)
        return response
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.exception(f"Unexpected error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of all LLM services."""
    health_status = {}
    for service_name, manager in router.managers.items():
        try:
            if manager.verify_gpu_setup() and manager.get_llm():
                health_status[service_name] = {
                    "status": "healthy",
                    "gpu_id": manager.gpu_id,
                }
            else:
                health_status[service_name] = {
                    "status": "unhealthy",
                    "gpu_id": manager.gpu_id,
                }
        except Exception as e:
            health_status[service_name] = {
                "status": "error",
                "message": str(e),
                "gpu_id": manager.gpu_id,
            }
    return health_status
