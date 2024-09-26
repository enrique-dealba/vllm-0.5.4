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
  --name vlm \
  -p 8888:8888 \
  -e MODEL_TYPE=LLM \
  -e LLM_MODEL_NAME="mistralai/Mistral-Small-Instruct-2409" \
  vllm:cuda11.8
```

To run Docker container with VLMs (Vision Language Models):

```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  --name vlm \
  -p 8888:8888 \
  -e MODEL_TYPE=LVM \
  -e LVM_MODEL_NAME="llava-hf/llava-1.5-7b-hf" \
  -e FIXED_IMAGE_URL="https://example.com/your-image.jpg" \
  vllm:cuda11.8

```

To query the FastAPI server:

```sh
curl -X POST "http://localhost:8888/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is the content of this image?\"}" | jq
```
