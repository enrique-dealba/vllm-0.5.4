# vllm-0.5.4
Testing vLLM 0.5.4

To build Docker image:

```sh
docker build -t vllm:cuda11.8 .
```

To run Docker container with LLMs:

```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --name llm_server \
  -p 8888:8888 \
  -e RUN_MODE=server \
  -e MODEL_TYPE=LLM \
  -e LLM_MODEL_NAME="mistralai/Mistral-Small-Instruct-2409" \
  -e HUGGING_FACE_HUB_TOKEN=<your-hugging-face-token> \
  vllm:cuda11.8

```

To run Docker container with VLMs (Vision Language Models):

```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --name vlm_server \
  -p 8888:8888 \
  -e RUN_MODE=server \
  -e MODEL_TYPE=VLM \
  -e VLM_MODEL_NAME="llava-hf/llava-1.5-7b-hf" \
  -e FIXED_IMAGE_URL="https://example.com/your-image.jpg" \
  vllm:cuda11.8

```

To run Docker container with Streamlit UI:
```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --name llm_ui \
  -p 8888:8888 \
  -e RUN_MODE=ui \
  -e MODEL_TYPE=LLM \
  -e LLM_MODEL_NAME="mistralai/Mistral-Small-Instruct-2409" \
  -e HUGGING_FACE_HUB_TOKEN=<your-hugging-face-token> \
  vllm:cuda11.8

```

To query the FastAPI server:

```sh
curl -X POST "http://localhost:8888/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is the content of this image?\"}" | jq
```

To query the Streamlit UI:

1. Since the container is running Streamlit on port `8888`, you need to set up SSH tunneling to access it from your local machine:
```sh
# Activate conda environment if not already active
conda activate llm  # or your preferred environment

# SSH tunneling
ssh -L 8888:localhost:8888 <your-username>@<machine>

```

2. Open your web browser and navigate to:
[http://localhost:8888](http://localhost:8888)
You should see the Streamlit UI where you can input queries and interact with the model.
