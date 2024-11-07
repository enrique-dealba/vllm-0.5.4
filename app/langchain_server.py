import logging
import time
import traceback
import uuid
from typing import List, Optional

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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


class DiscussionMessage(BaseModel):
    role: str
    message: str


class LLMResponse(BaseModel):
    response: str
    discussion: Optional[List[DiscussionMessage]] = []


class CollaborationResult(BaseModel):
    final_response: str
    discussion: List[DiscussionMessage]


async def process_request(query: str, debug_mode: bool = True) -> CollaborationResult:
    """Processes a single LLM generate request without collaboration."""
    discussion_messages: List[DiscussionMessage] = []
    debug_messages: List[DiscussionMessage] = []

    try:
        # Log the received query
        debug_messages.append(
            DiscussionMessage(
                role="debug", message=f"[Debug] Received query: '{query}'"
            )
        )

        # Generate response from LLM
        try:
            response = model_manager.get_llm().invoke(query)
            debug_messages.append(
                DiscussionMessage(
                    role="debug",
                    message=f"[Debug] LLM generated response ({len(response)} chars)",
                )
            )
            discussion_messages.append(
                DiscussionMessage(role=settings.SERVICE_ROLE, message=response)
            )
        except Exception as e:
            debug_messages.append(
                DiscussionMessage(
                    role="error",
                    message=f"[Error] Failed to generate response: {str(e)}",
                )
            )
            raise

        # Merge discussion and debug messages if debug mode is on
        all_messages = (
            debug_messages + discussion_messages if debug_mode else discussion_messages
        )

        return CollaborationResult(final_response=response, discussion=all_messages)

    except Exception as e:
        logger.error(f"Processing error: {str(e)}")
        debug_messages.append(
            DiscussionMessage(
                role="error",
                message=f"[Critical Error] Processing failed: {str(e)}\n{traceback.format_exc()}",
            )
        )
        raise


@app.post("/generate", response_model=LLMResponse)
async def generate(request: Request):
    """Generates a response based on the user query."""
    try:
        request_data = await request.json()
        logger.info(f"Generate request received: {request_data}")

        collaboration_result = await process_request(request_data.get("text", ""))
        return LLMResponse(
            response=collaboration_result.final_response,
            discussion=collaboration_result.discussion,
        )

    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        return LLMResponse(response=f"Error: {str(e)}", discussion=[])


# Add middleware for request tracking
@app.middleware("http")
async def track_requests(request: Request, call_next):
    try:
        request_id = str(uuid.uuid4())
        logger.info(f"[{request_id}] Incoming request to {request.url.path}")

        # Track timing
        start_time = time.time()
        response = await call_next(request)
        duration = time.time() - start_time

        logger.info(
            f"[{request_id}] Response status: {response.status_code}, Duration: {duration:.2f}s"
        )
        return response
    except Exception as e:
        logger.error(f"Middleware error: {e}")
        raise
