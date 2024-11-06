# Multi-LLM
Testing vLLM with multiple LLMs running on different GPUs

## Initial Model Pre-loading

Before running any other commands, pre-load the models to ensure efficient caching:
```sh
docker compose up llm1_preload
```

This step ensures that models are properly cached before starting the main services.

## Build Docker Image

```sh
docker build -t vllm:cuda11.8 .
```

## Setup and Deployment

### 1. Set Up Environment
Generate the necessary environment variables:
```sh
./setup.sh -h "your_huggingface_token" -l "your_langchain_api_key"
```

### 2. Deploy the Multi-LLM Services
Deploy all services, including the main server and LLM services:
```sh
./deploy_multi_llm.sh -h "your_huggingface_token" -l "your_langchain_api_key"
```

This will start:
- **LLM1** on port 8881 using GPU 0
- **LLM2** on port 8882 using GPU 1
- **Main Server** on port 8888, acting as the frontend interface and API aggregator

## Updated Docker Compose Configuration

The `docker-compose.yml` file defines:
- Two primary LLM services (`llm1` and `llm2`)
- A preload service for model caching
- A main server for handling user interactions
- Shared Hugging Face cache volume
- GPU device assignments
- Network configuration and health checks

## API Usage

### Main Server (Port 8888)
Interact with both LLM services through the main server:
```sh
curl -X POST http://localhost:8888/meta/generate \
    -H "Content-Type: application/json" \
    -d '{
          "text": "What is 7+8?"
        }'
```

### Individual LLM Endpoints

#### LLM1 (Port 8881)
```sh
curl -X POST http://localhost:8881/generate \
    -H "Content-Type: application/json" \
    -d '{
          "text": "Translate the following English text to French: Hello, how are you?"
        }'
```

#### LLM2 (Port 8882)
```sh
curl -X POST http://localhost:8882/generate \
    -H "Content-Type: application/json" \
    -d '{
          "text": "Summarize the following text: Machine learning enables computers to learn from data."
        }'
```

## Environment Variables

Key environment variables:
- `HUGGING_FACE_HUB_TOKEN`: Your Hugging Face authentication token
- `LANGCHAIN_API_KEY`: Your LangChain API key
- `LLM_MODEL_NAME`: Model name for each LLM service
- `CUDA_VISIBLE_DEVICES`: GPU device assignment
- `SERVICE_NAME`: Unique identifier for each service
- `LLM1_URL` and `LLM2_URL`: URLs for the LLM services used by the main server

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

## Health Checks and Dependency Management

Ensure that `llm1` and `llm2` are healthy before `main_server` starts:
```yaml
services:
  main_server:
    depends_on:
      llm1:
        condition: service_healthy
      llm2:
        condition: service_healthy
```

## Model Pre-loading

Before starting the main services, models are pre-loaded using the `llm1_preload` service to ensure faster startup times and proper caching:
```sh
docker compose up llm1_preload
```

## SSH Tunneling for UI Access

To access the frontend UI on your local machine:
```sh
ssh -L 8888:localhost:8888 your_username@your_server
```

Then, navigate to:
```
http://localhost:8888/
```
This will display the Multi-LLM Chat Interface for interacting with both LLMs.
