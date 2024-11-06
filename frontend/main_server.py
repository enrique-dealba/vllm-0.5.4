import asyncio
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

# Mount static files (HTML UI)
app.mount("/", StaticFiles(directory="static", html=True), name="static")


# Define request and response schemas
class MetaGenerateRequest(BaseModel):
    text: str


class MetaGenerateResponse(BaseModel):
    llm1_response: str
    llm2_response: str


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
            payload = {"text": request.text}

            # Send requests concurrently
            llm1_task = client.post(LLM1_URL, json=payload)
            llm2_task = client.post(LLM2_URL, json=payload)

            llm1_response, llm2_response = await asyncio.gather(llm1_task, llm2_task)

            # Raise exceptions for bad responses
            llm1_response.raise_for_status()
            llm2_response.raise_for_status()

            # Extract responses
            llm1_result = llm1_response.json().get("response", "")
            llm2_result = llm2_response.json().get("response", "")

            return MetaGenerateResponse(
                llm1_response=llm1_result, llm2_response=llm2_result
            )

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
