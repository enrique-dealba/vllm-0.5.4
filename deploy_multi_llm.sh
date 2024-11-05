#!/bin/bash
set -e  # Exit on any error

# Configuration
IMAGE_NAME="vllm:cuda11.8"
HF_TOKEN="your-huggingface-token"  # Replace or pass as argument
LANGCHAIN_API_KEY="your-langchain-key"  # Replace or pass as argument
SHARED_CACHE_DIR="$HOME/.cache/huggingface"

# Function to display script usage
usage() {
    echo "Usage: $0 [-h <hugging_face_token>] [-l <langchain_api_key>]"
    echo "  -h    Hugging Face token (optional if hardcoded)"
    echo "  -l    LangChain API key (optional if hardcoded)"
    exit 1
}

# Parse command line arguments
while getopts "h:l:" opt; do
    case $opt in
        h) HF_TOKEN="$OPTARG" ;;
        l) LANGCHAIN_API_KEY="$OPTARG" ;;
        ?) usage ;;
    esac
done

# Validate required tokens
if [ -z "$HF_TOKEN" ] || [ -z "$LANGCHAIN_API_KEY" ]; then
    echo "Error: Hugging Face token and LangChain API key are required"
    usage
fi

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        echo "Error: Docker is not running or not accessible"
        exit 1
    fi
}

# Function to build Docker image
build_docker_image() {
    echo "Building Docker image..."
    docker build -t "$IMAGE_NAME" .
}

# Function to check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        echo "Error: NVIDIA Docker runtime is not available"
        exit 1
    fi
}

# Function to clean up Docker containers
cleanup() {
    echo "Starting cleanup..."

    # Clean up multi-LLM service container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "multi-llm-service"; then
        echo "Removing existing multi-LLM container..."
        docker rm -f multi-llm-service 2>/dev/null || true
    fi

    # Clean up llm1 container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "llm1"; then
        echo "Removing existing llm1 container..."
        docker rm -f llm1 2>/dev/null || true
    fi

    # Clean up llm2 container if it exists
    if docker ps -a --format '{{.Names}}' | grep -q "llm2"; then
        echo "Removing existing llm2 container..."
        docker rm -f llm2 2>/dev/null || true
    fi

    echo "Cleanup completed."
}

# Function to deploy the multi-LLM service
deploy_multi_llm_service() {
    echo "Deploying multi-LLM service..."
    
    # First verify GPU availability
    if ! nvidia-smi > /dev/null 2>&1; then
        echo "Error: NVIDIA GPUs not accessible"
        exit 1
    fi
    
    echo "Available GPUs:"
    nvidia-smi -L
    
    container_id=$(docker run -d \
        --name "multi-llm-service" \
        -v "$SHARED_CACHE_DIR":/root/.cache/huggingface \
        --gpus '"device=0,1"' \
        --shm-size=32g \
        -p "8888:8888" \
        -e PORT="8888" \
        -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        -e LANGCHAIN_API_KEY="$LANGCHAIN_API_KEY" \
        -e PYTHONUNBUFFERED=1 \
        -e LOG_LEVEL=DEBUG \
        "$IMAGE_NAME")

    # Wait longer for initialization
    echo "Waiting for container initialization..."
    sleep 10

    # Verify GPU visibility inside container
    echo "Verifying GPU visibility in container:"
    docker exec multi-llm-service nvidia-smi || true

    # Check container status
    echo "Container Status:"
    docker ps -a --filter "name=multi-llm-service"
    
    echo -e "\nContainer Logs:"
    docker logs multi-llm-service

    # Verify container is running
    if ! docker ps --format '{{.Names}}' | grep -q "multi-llm-service"; then
        echo "Error: Failed to start multi-LLM service container"
        echo "Last container logs:"
        docker logs multi-llm-service --tail 50
        exit 1
    fi
}

# Function to deploy an LLM service
deploy_llm_service() {
    local service_name=$1
    local port=$2
    local gpu_device=$3

    echo "Deploying $service_name on GPU $gpu_device..."

    docker run -d \
        --name "$service_name" \
        -v "$SHARED_CACHE_DIR":/root/.cache/huggingface \
        --gpus "device=$gpu_device" \
        --shm-size=32g \
        -p "$port:8888" \
        -e PORT="8888" \
        -e SERVICE_NAME="$service_name" \
        -e CUDA_DEVICE="$gpu_device" \
        -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        -e LANGCHAIN_API_KEY="$LANGCHAIN_API_KEY" \
        -e PYTHONUNBUFFERED=1 \
        -e LOG_LEVEL=DEBUG \
        "$IMAGE_NAME"
}

# Main deployment process
main() {
    echo "Starting multi-LLM deployment..."

    # Initial checks
    check_docker
    check_nvidia_docker

    # Build the Docker image
    build_docker_image

    # Cleanup previous instances
    cleanup

    # Deploy the multi-LLM service
    # deploy_multi_llm_service

    # Multi-LLM:
    # Deploy llm1 on GPU 0
    deploy_llm_service "llm1" 8881 0

    # Deploy llm2 on GPU 1
    deploy_llm_service "llm2" 8882 1

    # Allow time for services to initialize
    max_attempts=10
    attempt=1

    while (( attempt <= max_attempts )); do
        echo "Health check (Attempt $attempt/$max_attempts)"

        # Health check for llm1
        if curl -s http://localhost:8881/health | grep -q '"status":"healthy"'; then
            echo "llm1 is healthy and running on port 8881"
        else
            if (( attempt == max_attempts )); then
                echo "Error: llm1 failed health check on port 8881 after $max_attempts attempts"
                docker logs llm1 --tail 50
            else
                echo "llm1 not yet healthy, retrying..."
            fi
        fi

        # Health check for llm2
        if curl -s http://localhost:8882/health | grep -q '"status":"healthy"'; then
            echo "llm2 is healthy and running on port 8882"
        else
            if (( attempt == max_attempts )); then
                echo "Error: llm2 failed health check on port 8882 after $max_attempts attempts"
                docker logs llm2 --tail 50
            else
                echo "llm2 not yet healthy, retrying..."
            fi
        fi

        # Exit loop if both services are healthy
        if curl -s http://localhost:8881/health | grep -q '"status":"healthy"' && curl -s http://localhost:8882/health | grep -q '"status":"healthy"'; then
            echo "Both llm1 and llm2 are healthy."
            break
        fi

        # Sleep before the next attempt if max attempts not reached
        if (( attempt < max_attempts )); then
            sleep 4
        fi
        ((attempt++))
    done

    echo "Deployment successful!"
    echo "Multi-LLM services available at: http://localhost:8881 and http://localhost:8882"

    # Show GPU status
    echo -e "\nCurrent GPU Status:"
    nvidia-smi
}


# Run the deployment
main

# Example usage
echo -e "\nExample API calls:"
echo 'curl -X POST "http://localhost:8888/llm1/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is 7+8?\"}"'
echo 'curl -X POST "http://localhost:8888/llm2/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is 7+8?\"}"'

# Health check example
echo -e "\nHealth check endpoint:"
echo 'curl http://localhost:8888/health'
