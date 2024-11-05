import logging

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.model import ModelManager

logger = logging.getLogger(__name__)

app = FastAPI(title=f"LangChain LLM API - {settings.SERVICE_NAME}", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize ModelManager
model_manager = ModelManager()


@app.post("/generate")
async def generate(request: Request):
    try:
        request_data = await request.json()
        logger.info(f"Received request: {request_data}")
        llm = model_manager.get_llm()
        if not llm:
            raise ValueError("LLM not initialized")

        response = llm.invoke(request_data["text"])
        return {"response": response}
    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    try:
        llm = model_manager.get_llm()
        status = "healthy" if llm else "unhealthy"
        return {
            "service_name": settings.SERVICE_NAME,
            "status": status,
            "gpu_id": settings.CUDA_DEVICE,
        }
    except Exception as e:
        logger.error(f"Health check error: {str(e)}", exc_info=True)
        return {"status": "error", "detail": str(e)}
