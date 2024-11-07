import logging
import time
import traceback
import uuid
from typing import List, Optional

import httpx
from fastapi import FastAPI, HTTPException, Request
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


async def collaborate_with_peer(
    query: str, debug_mode: bool = True
) -> CollaborationResult:
    """Enhanced collaboration with detailed debug tracking"""
    discussion_messages: List[DiscussionMessage] = []
    debug_messages: List[DiscussionMessage] = []

    try:
        # Initial debug info
        debug_messages.append(
            DiscussionMessage(
                role="debug",
                message=f"[Debug] Starting collaboration as {settings.SERVICE_ROLE}",
            )
        )
        debug_messages.append(
            DiscussionMessage(
                role="debug", message=f"[Debug] Query received: '{query}'"
            )
        )
        debug_messages.append(
            DiscussionMessage(
                role="debug",
                message=f"[Debug] Peer service URL: {settings.PEER_SERVICE_URL}",
            )
        )

        # Initial thoughts generation
        debug_messages.append(
            DiscussionMessage(
                role="debug",
                message=f"[Debug] {settings.SERVICE_ROLE} generating initial thoughts...",
            )
        )

        try:
            initial_prompt = (
                f"As {settings.SERVICE_ROLE}, analyzing: '{query}'. Initial thoughts:"
            )
            initial_response = model_manager.get_llm().invoke(initial_prompt)

            debug_messages.append(
                DiscussionMessage(
                    role="debug",
                    message=f"[Debug] Initial response generated ({len(initial_response)} chars)",
                )
            )
            debug_messages.append(
                DiscussionMessage(
                    role="debug",
                    message=f"[Debug] Initial response preview: {initial_response[:100]}...",
                )
            )

            discussion_messages.append(
                DiscussionMessage(role=settings.SERVICE_ROLE, message=initial_response)
            )
        except Exception as e:
            debug_messages.append(
                DiscussionMessage(
                    role="error",
                    message=f"[Error] Failed to generate initial thoughts: {str(e)}",
                )
            )
            raise

        # Peer collaboration
        debug_messages.append(
            DiscussionMessage(
                role="debug", message="[Debug] Initiating peer collaboration..."
            )
        )

        try:
            async with httpx.AsyncClient() as client:
                debug_messages.append(
                    DiscussionMessage(
                        role="debug",
                        message=f"[Debug] Sending request to peer at {settings.PEER_SERVICE_URL}",
                    )
                )

                peer_prompt = f"Considering my colleague's thoughts: '{initial_response}', what's your perspective on: '{query}'?"
                debug_messages.append(
                    DiscussionMessage(
                        role="debug",
                        message=f"[Debug] Peer prompt: {peer_prompt[:100]}...",
                    )
                )

                peer_response = await client.post(
                    f"{settings.PEER_SERVICE_URL}/generate", json={"text": peer_prompt}
                )

                debug_messages.append(
                    DiscussionMessage(
                        role="debug",
                        message=f"[Debug] Peer response status: {peer_response.status_code}",
                    )
                )

                if peer_response.status_code != 200:
                    debug_messages.append(
                        DiscussionMessage(
                            role="error",
                            message=f"[Error] Peer returned status {peer_response.status_code}",
                        )
                    )
                    raise HTTPException(
                        status_code=peer_response.status_code,
                        detail="Peer service error",
                    )

                peer_data = peer_response.json()
                debug_messages.append(
                    DiscussionMessage(
                        role="debug",
                        message=f"[Debug] Peer response keys: {list(peer_data.keys())}",
                    )
                )

                peer_message = peer_data["response"]
                debug_messages.append(
                    DiscussionMessage(
                        role="debug",
                        message=f"[Debug] Peer response preview: {peer_message[:100]}...",
                    )
                )

                discussion_messages.append(
                    DiscussionMessage(role="peer", message=peer_message)
                )

        except Exception as e:
            debug_messages.append(
                DiscussionMessage(
                    role="error", message=f"[Error] Peer collaboration failed: {str(e)}"
                )
            )
            raise

        # Final synthesis
        debug_messages.append(
            DiscussionMessage(
                role="debug", message="[Debug] Generating final synthesis..."
            )
        )

        try:
            final_prompt = (
                f"After peer discussion on '{query}', synthesizing final response..."
            )
            final_response = model_manager.get_llm().invoke(final_prompt)

            debug_messages.append(
                DiscussionMessage(
                    role="debug",
                    message=f"[Debug] Final response generated ({len(final_response)} chars)",
                )
            )
            debug_messages.append(
                DiscussionMessage(
                    role="debug",
                    message=f"[Debug] Final response preview: {final_response[:100]}...",
                )
            )

            discussion_messages.append(
                DiscussionMessage(role=settings.SERVICE_ROLE, message=final_response)
            )

        except Exception as e:
            debug_messages.append(
                DiscussionMessage(
                    role="error", message=f"[Error] Final synthesis failed: {str(e)}"
                )
            )
            raise

        # Merge discussion and debug messages if debug mode is on
        all_messages = (
            debug_messages + discussion_messages if debug_mode else discussion_messages
        )

        return CollaborationResult(
            final_response=final_response, discussion=all_messages
        )

    except Exception as e:
        logger.error(f"Collaboration error: {str(e)}")
        debug_messages.append(
            DiscussionMessage(
                role="error",
                message=f"[Critical Error] Collaboration failed: {str(e)}\n{traceback.format_exc()}",
            )
        )
        raise


@app.post("/generate", response_model=LLMResponse)
async def generate(request: Request):
    """Main generate endpoint using collaboration"""
    debug_messages: List[DiscussionMessage] = []

    try:
        request_data = await request.json()
        logger.info(f"Generate request received: {request_data}")

        debug_messages.append(
            DiscussionMessage(
                role="debug",
                message=f"[Debug] Generate endpoint received request: {request_data}",
            )
        )

        # Get collaborative response
        try:
            collaboration_result = await collaborate_with_peer(request_data["text"])
            debug_messages.append(
                DiscussionMessage(
                    role="debug",
                    message=f"[Debug] Collaboration successful. Result length: {len(collaboration_result.discussion)} messages",
                )
            )
        except Exception as e:
            debug_messages.append(
                DiscussionMessage(
                    role="error", message=f"[Error] Collaboration failed: {str(e)}"
                )
            )
            raise

        return LLMResponse(
            response=collaboration_result.final_response,
            discussion=collaboration_result.discussion,
        )

    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        debug_messages.append(
            DiscussionMessage(
                role="error",
                message=f"[Critical Error] Generation failed: {str(e)}\n{traceback.format_exc()}",
            )
        )
        return LLMResponse(response=f"Error: {str(e)}", discussion=debug_messages)


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
