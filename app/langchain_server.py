import logging
from typing import Dict

import httpx
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


async def collaborate_with_peer(query: str) -> Dict:
    """Have a discussion with the peer LLM about the query."""
    discussion_messages = []
    debug_messages = []

    try:
        debug_messages.append(
            {
                "role": "debug",
                "message": f"[Debug] Starting collaboration for query: '{query}' with role: {settings.SERVICE_ROLE}",
            }
        )

        async with httpx.AsyncClient() as client:
            # Initial thoughts from this LLM
            debug_messages.append(
                {
                    "role": "debug",
                    "message": f"[Debug] {settings.SERVICE_ROLE} generating initial thoughts...",
                }
            )

            initial_thought = (
                f"Let me think about '{query}'. Here are my initial thoughts..."
            )
            try:
                my_response = model_manager.get_llm().invoke(initial_thought)
                debug_messages.append(
                    {
                        "role": "debug",
                        "message": f"[Debug] {settings.SERVICE_ROLE} initial response generated: {my_response[:100]}...",
                    }
                )
            except Exception as e:
                debug_messages.append(
                    {
                        "role": "error",
                        "message": f"[Error] Failed to generate initial thoughts: {str(e)}",
                    }
                )
                raise

            discussion_messages.append(
                {"role": settings.SERVICE_ROLE, "message": my_response}
            )

            # Send to peer for discussion
            debug_messages.append(
                {
                    "role": "debug",
                    "message": f"[Debug] Sending to peer at {settings.PEER_SERVICE_URL}",
                }
            )

            peer_prompt = f"My colleague thinks: '{my_response}'. What are your thoughts on '{query}'?"
            try:
                peer_response = await client.post(
                    f"{settings.PEER_SERVICE_URL}/generate", json={"text": peer_prompt}
                )
                debug_messages.append(
                    {
                        "role": "debug",
                        "message": f"[Debug] Peer response status: {peer_response.status_code}",
                    }
                )

                peer_data = peer_response.json()
                debug_messages.append(
                    {
                        "role": "debug",
                        "message": f"[Debug] Peer response keys: {list(peer_data.keys())}",
                    }
                )
            except Exception as e:
                debug_messages.append(
                    {
                        "role": "error",
                        "message": f"[Error] Failed to get peer response: {str(e)}",
                    }
                )
                raise

            discussion_messages.append(
                {"role": "peer", "message": peer_data["response"]}
            )

            # Final thoughts after discussion
            debug_messages.append(
                {"role": "debug", "message": "[Debug] Generating final thoughts..."}
            )

            final_thought = f"After discussing with my colleague who said '{peer_data['response']}', here's my final answer..."
            try:
                final_response = model_manager.get_llm().invoke(final_thought)
                debug_messages.append(
                    {
                        "role": "debug",
                        "message": f"[Debug] Final response generated: {final_response[:100]}...",
                    }
                )
            except Exception as e:
                debug_messages.append(
                    {
                        "role": "error",
                        "message": f"[Error] Failed to generate final response: {str(e)}",
                    }
                )
                raise

            discussion_messages.append(
                {"role": settings.SERVICE_ROLE, "message": final_response}
            )

            # Merge discussion and debug messages
            all_messages = debug_messages + discussion_messages

            return {
                "discussion": all_messages,
                "final_response": final_response,
                "debug_log": debug_messages,  # Separate debug log if needed
            }

    except Exception as e:
        logger.error(f"Collaboration error: {str(e)}")
        debug_messages.append(
            {
                "role": "error",
                "message": f"[Critical Error] Collaboration failed: {str(e)}",
            }
        )
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: Request):
    try:
        request_data = await request.json()
        logger.info(f"Received request: {request_data}")

        debug_messages = [
            {
                "role": "debug",
                "message": f"[Debug] Generate endpoint received request: {request_data}",
            }
        ]

        try:
            # Get collaborative response
            collaboration_result = await collaborate_with_peer(request_data["text"])
            debug_messages.append(
                {
                    "role": "debug",
                    "message": f"[Debug] Collaboration complete. Result keys: {list(collaboration_result.keys())}",
                }
            )
        except Exception as e:
            debug_messages.append(
                {"role": "error", "message": f"[Error] Collaboration failed: {str(e)}"}
            )
            raise

        return {
            "response": collaboration_result["final_response"],
            "discussion": collaboration_result["discussion"],
            "debug_log": debug_messages + collaboration_result.get("debug_log", []),
        }

    except Exception as e:
        logger.error(f"Generation error: {str(e)}", exc_info=True)
        return {
            "response": f"Error: {str(e)}",
            "discussion": debug_messages,
            "debug_log": debug_messages,
        }


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
