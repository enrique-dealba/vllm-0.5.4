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
    debug_discussion = []
    try:
        async with httpx.AsyncClient() as client:
            # Debug: Log initial request
            debug_discussion.append(
                {
                    "role": "debug",
                    "message": f"[Debug] Received user query: {request.text}",
                }
            )

            # Get responses from both LLMs
            debug_discussion.append(
                {
                    "role": "debug",
                    "message": f"[Debug] Attempting to contact LLM1 at {LLM1_URL}",
                }
            )
            llm1_response = await client.post(LLM1_URL, json={"text": request.text})

            debug_discussion.append(
                {
                    "role": "debug",
                    "message": f"[Debug] Attempting to contact LLM2 at {LLM2_URL}",
                }
            )
            llm2_response = await client.post(LLM2_URL, json={"text": request.text})

            llm1_data = llm1_response.json()
            llm2_data = llm2_response.json()

            # Debug: Log response data structure
            debug_discussion.append(
                {
                    "role": "debug",
                    "message": f"[Debug] LLM1 response keys: {list(llm1_data.keys())}",
                }
            )
            debug_discussion.append(
                {
                    "role": "debug",
                    "message": f"[Debug] LLM2 response keys: {list(llm2_data.keys())}",
                }
            )

            # Extract the discussion between LLMs
            llm_discussion = []
            if "discussion" in llm1_data:
                debug_discussion.append(
                    {
                        "role": "debug",
                        "message": "[Debug] Found discussion in LLM1 response",
                    }
                )
                llm_discussion.extend(llm1_data["discussion"])
            else:
                debug_discussion.append(
                    {
                        "role": "debug",
                        "message": "[Debug] No discussion found in LLM1 response",
                    }
                )

            if "discussion" in llm2_data:
                debug_discussion.append(
                    {
                        "role": "debug",
                        "message": "[Debug] Found discussion in LLM2 response",
                    }
                )
                llm_discussion.extend(llm2_data["discussion"])
            else:
                debug_discussion.append(
                    {
                        "role": "debug",
                        "message": "[Debug] No discussion found in LLM2 response",
                    }
                )

            # If no real discussion, use debug info
            if not llm_discussion:
                debug_discussion.append(
                    {
                        "role": "debug",
                        "message": "[Debug] No LLM discussion found, using debug messages",
                    }
                )
                llm_discussion = debug_discussion

            return {
                "llm1_response": llm1_data["response"],
                "llm2_response": llm2_data["response"],
                "llm_discussion": llm_discussion,
            }

    except httpx.HTTPError as http_err:
        logger.error(f"HTTP error occurred: {http_err}")
        debug_discussion.append(
            {"role": "error", "message": f"[Error] HTTP error: {str(http_err)}"}
        )
        raise HTTPException(
            status_code=500, detail="Error communicating with LLM services"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        debug_discussion.append(
            {"role": "error", "message": f"[Error] Unexpected error: {str(e)}"}
        )
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/health")
async def health_check():
    return {"service": "main_server", "status": "healthy"}


# Mount static files AFTER defining API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")
