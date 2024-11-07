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
    llm_discussion: list


# Environment variables for LLM service URLs
LLM1_URL = os.getenv("LLM1_URL", "http://llm1:8888/generate")
LLM2_URL = os.getenv("LLM2_URL", "http://llm2:8888/generate")


@app.post("/meta/generate", response_model=MetaGenerateResponse)
async def meta_generate(request: MetaGenerateRequest):
    """Orchestrates collaboration between LLM1 and LLM2."""
    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Send initial query to LLM1
            llm1_prompt = f"As LLM1, analyzing: '{request.text}'. Initial thoughts:"
            llm1_resp = await client.post(LLM1_URL, json={"text": llm1_prompt})
            if llm1_resp.status_code != 200:
                raise HTTPException(
                    status_code=llm1_resp.status_code, detail="LLM1 service error"
                )
            llm1_data = llm1_resp.json()
            llm1_response = llm1_data.get("response", "")

            # Step 2: Send LLM1's response to LLM2
            llm2_prompt = f"Considering my colleague's thoughts: '{llm1_response}', what's your perspective on: '{request.text}'?"
            llm2_resp = await client.post(LLM2_URL, json={"text": llm2_prompt})
            if llm2_resp.status_code != 200:
                raise HTTPException(
                    status_code=llm2_resp.status_code, detail="LLM2 service error"
                )
            llm2_data = llm2_resp.json()
            llm2_response = llm2_data.get("response", "")

            # Step 3: Send LLM2's response back to LLM1 for final synthesis
            final_prompt = f"After peer discussion on '{request.text}', synthesizing final response: '{llm2_response}'"
            llm1_final_resp = await client.post(LLM1_URL, json={"text": final_prompt})
            if llm1_final_resp.status_code != 200:
                raise HTTPException(
                    status_code=llm1_final_resp.status_code,
                    detail="LLM1 final synthesis error",
                )
            llm1_final_data = llm1_final_resp.json()
            llm1_final_response = llm1_final_data.get("response", "")

            # Construct the discussion thread
            llm_discussion = [
                {"role": "LLM1", "message": llm1_response},
                {"role": "LLM2", "message": llm2_response},
                {"role": "LLM1", "message": llm1_final_response},
            ]

            return {
                "llm1_response": llm1_final_response,
                "llm2_response": llm2_response,
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
