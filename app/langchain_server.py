import logging

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.router import LLMRouter

logger = logging.getLogger(__name__)

app = FastAPI(title="LangChain LLM API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check available GPUs
n_gpus = torch.cuda.device_count()
logger.info(f"Number of available GPUs: {n_gpus}")

# Initialize the router with service names and GPU IDs
# Both services will use GPU 0 since that's all we have
router = LLMRouter({"llm1": 0, "llm2": 1})  # Both using GPU 0


@app.post("/{service_name}/generate")
async def generate(service_name: str, request: Request):
    try:
        request_data = await request.json()
        logger.info(f"Received request for {service_name}: {request_data}")
        response = await router.forward_request(service_name, request_data)
        return response
    except ValueError as ve:
        logger.error(f"ValueError: {ve}")
        raise HTTPException(status_code=400, detail=str(ve))
    except Exception as e:
        logger.error(f"Unexpected error during generation: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check the health status of all LLM services."""
    try:
        health_status = {"gpu_count": n_gpus, "services": {}}
        for service_name, manager in router.managers.items():
            try:
                if manager.verify_gpu_setup() and manager.get_llm():
                    health_status["services"][service_name] = {
                        "status": "healthy",
                        "gpu_id": manager.gpu_id,
                    }
                else:
                    health_status["services"][service_name] = {
                        "status": "unhealthy",
                        "gpu_id": manager.gpu_id,
                    }
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "error",
                    "message": str(e),
                    "gpu_id": manager.gpu_id,
                }
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return {"status": "error", "detail": str(e)}
