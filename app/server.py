import time

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from huggingface_hub import login

from app.config import settings

app = FastAPI()

# Authenticate with Hugging Face Hub
if settings.HUGGING_FACE_HUB_TOKEN:
    try:
        login(token=settings.HUGGING_FACE_HUB_TOKEN)
        print("Successfully logged in to HuggingFace Hub.")
    except Exception as e:
        print(f"Failed to authenticate with HuggingFace Hub: {e}")
        llm = None
        lvm = None
else:
    print("HUGGING_FACE_HUB_TOKEN not provided.")
    llm = None
    lvm = None

# Initialize the appropriate model based on MODEL_TYPE
if settings.MODEL_TYPE.upper() == "LLM":
    try:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=settings.LLM_MODEL_NAME,
            # Add additional initialization parameters if needed
        )
        print(f"LLM model '{settings.LLM_MODEL_NAME}' initialized successfully.")
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        llm = None
elif settings.MODEL_TYPE.upper() == "VLM":
    try:
        from vllm import LLM as VLLM_LLM

        from app.utils import load_image

        vlm = VLLM_LLM(
            model=settings.VLM_MODEL_NAME,
            # Add additional VLM-specific parameters if needed
        )
        image = load_image(url=settings.FIXED_IMAGE_URL)
        print(
            f"VLM model '{settings.VLM_MODEL_NAME}' initialized successfully with image '{settings.FIXED_IMAGE_URL}'."
        )
    except Exception as e:
        print(f"Error initializing VLM: {e}")
        vlm = None
else:
    raise ValueError(
        f"Invalid MODEL_TYPE: {settings.MODEL_TYPE}. Must be 'LLM' or 'VLM'."
    )


@app.post("/generate")
async def generate_response(request: Request):
    try:
        request_data = await request.json()
        query = request_data.get("text")

        if not query:
            raise HTTPException(
                status_code=400, detail="No text provided for generation."
            )

        start_time = time.time()

        if settings.MODEL_TYPE.upper() == "LLM":
            if llm is None:
                raise HTTPException(status_code=503, detail="LLM is not available.")

            sampling_params = SamplingParams(
                temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS
            )
            outputs = llm.generate([query], sampling_params=sampling_params)
            generated_text = outputs[0].outputs[0].text

        elif settings.MODEL_TYPE.upper() == "VLM":
            if vlm is None:
                raise HTTPException(status_code=503, detail="VLM is not available.")

            prompt = f"USER: <image>\n{query}\nASSISTANT:"
            sampling_params = SamplingParams(
                temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS
            )
            outputs = vlm.generate(
                [
                    {
                        "prompt": prompt,
                        "multi_modal_data": {"image": image},
                    }
                ],
                sampling_params,
            )
            generated_text = outputs[0].outputs[0].text

        else:
            raise HTTPException(
                status_code=500, detail="Invalid MODEL_TYPE configuration."
            )

        end_time = time.time()
        execution_time = end_time - start_time

        return JSONResponse(
            {
                "response": generated_text,
                "execution_time_seconds": round(execution_time, 4),
            }
        )

    except HTTPException as he:
        raise he
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Check health status of the model service."""
    if settings.MODEL_TYPE.upper() == "LLM":
        if llm is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "LLM is not initialized. GPU may not be available.",
                },
            )
    elif settings.MODEL_TYPE.upper() == "VLM":
        if vlm is None:
            return JSONResponse(
                status_code=503,
                content={
                    "status": "unhealthy",
                    "message": "VLM is not initialized. GPU may not be available.",
                },
            )
    else:
        return JSONResponse(
            status_code=500,
            content={
                "status": "invalid",
                "message": "Invalid MODEL_TYPE configuration.",
            },
        )

    return JSONResponse(
        {"status": "healthy", "message": "Model is initialized and ready."}
    )
