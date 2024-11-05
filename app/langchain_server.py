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

# Check GPU availability
n_gpus = torch.cuda.device_count()
logger.info(f"Number of available GPUs: {n_gpus}")
for i in range(n_gpus):
    logger.info(f"GPU {i}: {torch.cuda.get_device_name(i)}")

# Initialize router with appropriate GPU assignments
router = LLMRouter(
    {
        "llm1": 0,  # First service on GPU 0
        "llm2": 1,  # Second service on GPU 1
    }
)


@app.post("/{service_name}/generate")
async def generate(service_name: str, request: Request):
    try:
        request_data = await request.json()
        logger.info(f"Received request for {service_name}: {request_data}")
        response = await router.forward_request(service_name, request_data)
        return response
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        health_status = {"gpu_count": n_gpus, "services": {}}
        for service_name, manager in router.managers.items():
            try:
                gpu_memory = torch.cuda.memory_allocated(manager.gpu_id)
                health_status["services"][service_name] = {
                    "status": "healthy" if manager.get_llm() else "unhealthy",
                    "gpu_id": manager.gpu_id,
                    "gpu_memory_used": f"{gpu_memory/1024**3:.2f}GB",
                }
            except Exception as e:
                health_status["services"][service_name] = {
                    "status": "error",
                    "message": str(e),
                    "gpu_id": manager.gpu_id,
                }
        return health_status
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return {"status": "error", "detail": str(e)}
