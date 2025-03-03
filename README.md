## Offline LLMs for SDA Objectives

Build a Docker image using:
```sh
docker build -t vllm:cuda11.8 .
```

To start quickly with the MockLLM (no GPU required):
```sh
docker run -p 8888:8888 -e RUN_MODE=server -e USE_MOCK_LLM=true vllm:cuda11.8
```

Then test with:
```sh
curl http://localhost:8888/test-mock
```

For GPU usage, mount a local Huggging Face cache, enable GPU support, and specify env variables:
```sh
docker run \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  --gpus all \
  -p 8888:8888 \
  -e RUN_MODE=server \
  -e MODEL_TYPE=LLM \
  -e LLM_MODEL_NAME="microsoft/Phi-3.5-mini-instruct" \
  -e IS_MISTRAL=false \
  -e LANGCHAIN_PROJECT="demo-20250206" \
  -e HUGGING_FACE_HUB_TOKEN=... \
  -e LANGCHAIN_API_KEY=... \
  vllm:cuda11.8
```

Then send queries like:
```sh
curl -X POST -H "Content-Type: application/json" \
-d '{"text":"Track object 23456 with sensor RME01..."}' \
http://localhost:8888/generate_full_objective
```

You’ll recieve structured JSON for use in MACHINA. Adjust config settings in `app/config.py` as desired.
