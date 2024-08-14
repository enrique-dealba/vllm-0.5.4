from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from vllm import LLM, SamplingParams

from app.config import settings
from app.utils import load_image

app = FastAPI()

# Initialize the LLM
llm = LLM(model=settings.MODEL_NAME)

# Load and cache the image
image = load_image()


@app.post("/generate")
async def generate_response(request: Request):
    try:
        request_data = await request.json()
        query = request_data.get("text")
        prompt = f"USER: <image>\n{query}\nASSISTANT:"
        outputs = llm.generate(
            [
                {
                    "prompt": prompt,
                    "multi_modal_data": {"image": image},
                }
            ],
            SamplingParams(
                temperature=settings.TEMPERATURE, max_tokens=settings.MAX_TOKENS
            ),
        )
        generated_text = outputs[0].outputs[0].text
        return JSONResponse({"response": generated_text})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=settings.HOST, port=settings.PORT)
