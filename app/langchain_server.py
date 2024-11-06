import logging

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
    try:
        discussion_messages = []
        async with httpx.AsyncClient() as client:
            # Initial thoughts from this LLM
            initial_thought = (
                f"Let me think about '{query}'. Here are my initial thoughts..."
            )
            my_response = model_manager.get_llm().invoke(initial_thought)
            discussion_messages.append(
                {"role": settings.SERVICE_ROLE, "message": my_response}
            )

            # Send to peer for discussion
            peer_prompt = f"My colleague thinks: '{my_response}'. What are your thoughts on '{query}'?"
            peer_response = await client.post(
                f"{settings.PEER_SERVICE_URL}/generate", json={"text": peer_prompt}
            )
            peer_data = peer_response.json()
            discussion_messages.append(
                {"role": "peer", "message": peer_data["response"]}
            )

            # Final thoughts after discussion
            final_thought = f"After discussing with my colleague who said '{peer_data['response']}', here's my final answer..."
            final_response = model_manager.get_llm().invoke(final_thought)
            discussion_messages.append(
                {"role": settings.SERVICE_ROLE, "message": final_response}
            )

            return {"discussion": discussion_messages, "final_response": final_response}
    except Exception as e:
        logger.error(f"Collaboration error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/generate")
async def generate(request: Request):
    try:
        request_data = await request.json()
        logger.info(f"Received request: {request_data}")

        # Get collaborative response
        collaboration_result = await collaborate_with_peer(request_data["text"])

        return {
            "response": collaboration_result["final_response"],
            "discussion": collaboration_result["discussion"],
        }
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
