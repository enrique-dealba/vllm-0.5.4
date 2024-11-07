import logging
import os
import uuid

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

# Define the number of retries
MAX_RETRIES = 3


@app.post("/meta/generate", response_model=MetaGenerateResponse)
async def meta_generate(request: MetaGenerateRequest):
    """Orchestrates collaboration between LLM1 and LLM2 with retry mechanism."""
    request_id = str(uuid.uuid4())
    logger.info(f"[{request_id}] Received user query: '{request.text}'")

    try:
        async with httpx.AsyncClient() as client:
            # Step 1: Send initial query to LLM1
            llm1_prompt = f"As LLM1, analyze the following: '{request.text}'. Give you initial thoughts:"
            logger.info(f"[{request_id}] Sending initial prompt to LLM1: {llm1_prompt}")
            llm1_resp = await client.post(LLM1_URL, json={"text": llm1_prompt})

            if llm1_resp.status_code != 200:
                logger.error(
                    f"[{request_id}] LLM1 responded with status {llm1_resp.status_code}"
                )
                raise HTTPException(
                    status_code=llm1_resp.status_code, detail="LLM1 service error"
                )

            llm1_data = llm1_resp.json()
            llm1_response = llm1_data.get("response", "").strip()
            logger.info(f"[{request_id}] Received response from LLM1: {llm1_response}")

            # Step 2: Send LLM1's response to LLM2
            llm2_prompt = (
                f"You are LLM2. Now, considering LLM1's thoughts: '{llm1_response}', "
                f"what is your perspective on: '{request.text}'?"
            )
            logger.info(f"[{request_id}] Sending prompt to LLM2: {llm2_prompt}")
            llm2_resp = await client.post(LLM2_URL, json={"text": llm2_prompt})

            if llm2_resp.status_code != 200:
                logger.error(
                    f"[{request_id}] LLM2 responded with status {llm2_resp.status_code}"
                )
                raise HTTPException(
                    status_code=llm2_resp.status_code, detail="LLM2 service error"
                )

            llm2_data = llm2_resp.json()
            llm2_response = llm2_data.get("response", "").strip()
            logger.info(f"[{request_id}] Received response from LLM2: {llm2_response}")

            # Step 3: Send LLM2's response back to LLM1 for final synthesis with retries
            final_prompt = (
                f"Based on the following discussion:\n"
                f"'LLM1: {llm1_response}'\n"
                f"'LLM2: {llm2_response}',\n"
                f"please provide a final synthesized response to the question: {request.text}:"
            )
            logger.info(f"[{request_id}] Sending final prompt to LLM1: {final_prompt}")

            for attempt in range(1, MAX_RETRIES + 1):
                final_prompt_llm1 = f"You are LLM1. {final_prompt}"
                llm1_final_resp = await client.post(
                    LLM1_URL, json={"text": final_prompt_llm1}
                )

                if llm1_final_resp.status_code != 200:
                    logger.error(
                        f"[{request_id}] LLM1 final synthesis attempt {attempt} failed with status {llm1_final_resp.status_code}"
                    )
                    if attempt == MAX_RETRIES:
                        raise HTTPException(
                            status_code=llm1_final_resp.status_code,
                            detail="LLM1 final synthesis error",
                        )
                    continue

                llm1_final_data = llm1_final_resp.json()
                llm1_final_response = llm1_final_data.get("response", "").strip()
                logger.info(
                    f"[{request_id}] Received final response [after {attempt} attempts] from LLM1: {llm1_final_response}"
                )

                if llm1_final_response:
                    break
                else:
                    logger.warning(
                        f"[{request_id}] LLM1 final synthesis attempt {attempt} returned empty response"
                    )
                    if attempt == MAX_RETRIES:
                        llm1_final_response = "I'm sorry, I couldn't synthesize a final response at this time."

            for attempt in range(1, MAX_RETRIES + 1):
                personality_llm2 = "you are very critical and paranoid"
                final_prompt_llm2 = (
                    f"You are LLM2, and {personality_llm2}. {final_prompt}"
                )
                llm2_final_resp = await client.post(
                    LLM2_URL, json={"text": final_prompt_llm2}
                )

                if llm2_final_resp.status_code != 200:
                    logger.error(
                        f"[{request_id}] LLM2 final synthesis attempt {attempt} failed with status {llm2_final_resp.status_code}"
                    )
                    if attempt == MAX_RETRIES:
                        raise HTTPException(
                            status_code=llm2_final_resp.status_code,
                            detail="LLM2 final synthesis error",
                        )
                    continue

                llm2_final_data = llm2_final_resp.json()
                llm2_final_response = llm2_final_data.get("response", "").strip()
                logger.info(
                    f"[{request_id}] Received final response [after {attempt} attempts] from LLM2: {llm2_final_response}"
                )

                if llm2_final_response:
                    break
                else:
                    logger.warning(
                        f"[{request_id}] LLM2 final synthesis attempt {attempt} returned empty response"
                    )
                    if attempt == MAX_RETRIES:
                        llm2_final_response = "I'm sorry, I couldn't synthesize a final response at this time."

            # Construct the discussion thread
            llm_discussion = [
                {"role": "LLM1", "message": llm1_response},
                {"role": "LLM2", "message": llm2_response},
                # {"role": "LLM1", "message": llm1_final_response}
            ]

            return {
                "llm1_response": llm1_final_response,
                "llm2_response": llm2_final_response,
                "llm_discussion": llm_discussion,
            }

    except httpx.HTTPError as http_err:
        logger.error(f"[{request_id}] HTTP error occurred: {http_err}")
        raise HTTPException(
            status_code=500, detail="Error communicating with LLM services"
        )
    except Exception as e:
        logger.error(f"[{request_id}] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An unexpected error occurred")


@app.get("/health")
async def health_check():
    return {"service": "main_server", "status": "healthy"}


# Mount static files AFTER defining API routes
app.mount("/", StaticFiles(directory="static", html=True), name="static")
