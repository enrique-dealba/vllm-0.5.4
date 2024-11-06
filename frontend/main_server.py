import logging
import os

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main_server")

app = FastAPI(title="Multi-LLM Frontend API", version="1.0.0")

# CORS settings
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Adjust as needed for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Define request and response schemas
class MetaGenerateRequest(BaseModel):
    text: str


class MetaGenerateResponse(BaseModel):
    llm1_response: str
    llm2_response: str
    llm_discussion: list = [
        {
            "role": "llm1",
            "message": "[Mock] LLM1: Let me analyze this query in detail...",
        },
        {
            "role": "llm2",
            "message": "[Mock] LLM2: I agree, here's my additional perspective...",
        },
        {
            "role": "llm1",
            "message": "[Mock] LLM1: Great points. Let's finalize our responses...",
        },
    ]


# Environment variables for LLM service URLs
LLM1_URL = os.getenv("LLM1_URL", "http://llm1:8888/generate")
LLM2_URL = os.getenv("LLM2_URL", "http://llm2:8888/generate")


@app.post("/meta/generate", response_model=MetaGenerateResponse)
async def meta_generate(request: MetaGenerateRequest):
    """Receives a user query and forwards it to both LLM1 and LLM2.
    Returns the aggregated responses.
    """
    try:
        async with httpx.AsyncClient() as client:
            # Get responses from both LLMs
            llm1_response = await client.post(LLM1_URL, json={"text": request.text})
            llm2_response = await client.post(LLM2_URL, json={"text": request.text})

            llm1_data = llm1_response.json()
            llm2_data = llm2_response.json()

            # Extract the discussion between LLMs
            llm_discussion = []
            if "discussion" in llm1_data:
                llm_discussion.extend(llm1_data["discussion"])
            if "discussion" in llm2_data:
                llm_discussion.extend(llm2_data["discussion"])

            return {
                "llm1_response": llm1_data["response"],
                "llm2_response": llm2_data["response"],
                "llm_discussion": llm_discussion,
            }

    except httpx.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        raise HTTPException(
            status_code=500, detail="Error communicating with LLM services"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/health")
async def health_check():
    return {"service": "main_server", "status": "healthy"}


# Mount static files AFTER defining API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")
