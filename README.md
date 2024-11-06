# Multi-LLM
Testing vLLM with multiple LLMs running on different GPUs

## Initial Model Pre-loading

Before running any other commands, pre-load the models:
```sh
docker compose up llm1_preload
```

This ensures models are properly cached before starting the main services.

## Build Docker Image

```sh
docker build -t vllm:cuda11.8 .
```

## Setup and Deployment

1. Set up environment:
```sh
./setup.sh -h "your_huggingface_token" -l "your_langchain_api_key"
```

2. Deploy the multi-LLM services:
```sh
./deploy_multi_llm.sh -h "your_huggingface_token" -l "your_langchain_api_key"
```

This will start two LLM services:
- LLM1 on port 8881 using GPU 0
- LLM2 on port 8882 using GPU 1

## Docker Compose Configuration

The system uses a `docker-compose.yml` file that defines:
- Two main LLM services (llm1 and llm2)
- A preload service for model caching
- Shared Hugging Face cache volume
- GPU device assignments
- Network configuration

## API Usage

### LLM1 (Port 8881)
```sh
curl -X POST http://localhost:8881/generate \
    -H "Content-Type: application/json" \
    -d '{
          "text": "Translate the following English text to French: Hello, how are you?"
        }'
```

### LLM2 (Port 8882)
```sh
curl -X POST http://localhost:8882/generate \
    -H "Content-Type: application/json" \
    -d '{
          "text": "Summarize the following text: Machine learning enables computers to learn from data."
        }'
```

## Environment Variables

Key environment variables in the configuration:
- `HUGGING_FACE_HUB_TOKEN`: Your Hugging Face authentication token
- `LANGCHAIN_API_KEY`: Your Langchain API key
- `LLM_MODEL_NAME`: Model name for each LLM service
- `CUDA_VISIBLE_DEVICES`: GPU device assignment
- `SERVICE_NAME`: Unique identifier for each service

## Volume Mounts

The system mounts the local Hugging Face cache to ensure efficient model loading:
```yaml
volumes:
  - ~/.cache/huggingface:/root/.cache/huggingface
```

## Network Configuration

Services communicate through a bridge network defined in Docker Compose:
```yaml
networks:
  llm-network:
    driver: bridge
```

## Model Pre-loading

Before starting the main services, models are pre-loaded using the `llm1_preload` service to ensure faster startup times and proper caching.
```sh
docker compose up llm1_preload
```
