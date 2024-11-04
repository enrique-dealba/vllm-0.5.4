#!/bin/bash
set -e  # Exit on any error

# Configuration
IMAGE_NAME="vllm:cuda11.8"
HF_TOKEN="your-huggingface-token"  # Replace or pass as argument
LANGCHAIN_API_KEY="your-langchain-key"  # Replace or pass as argument
SHARED_CACHE_DIR="$HOME/.cache/huggingface"
NETWORK_NAME="llm-network"
MODEL_NAME="mistralai/Mistral-Small-Instruct-2409"

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

# Function to check if NVIDIA Docker runtime is available
check_nvidia_docker() {
    if ! docker run --rm --gpus all nvidia/cuda:11.8.0-base-ubuntu22.04 nvidia-smi > /dev/null 2>&1; then
        echo "Error: NVIDIA Docker runtime is not available"
        exit 1
    fi
}

# Function to clean up Docker containers and networks
cleanup() {
    echo "Starting cleanup..."
    
    # Clean up containers if they exist
    if docker ps -a --format '{{.Names}}' | grep -q "llm[12]"; then
        echo "Removing existing LLM containers..."
        docker rm -f llm1 llm2 2>/dev/null || true
    fi
    
    # Clean up network if it exists
    if docker network ls --format '{{.Name}}' | grep -q "^${NETWORK_NAME}$"; then
        echo "Removing Docker network: ${NETWORK_NAME}"
        docker network rm "${NETWORK_NAME}" 2>/dev/null || true
    fi
    
    echo "Cleanup completed."
}

# Function to create Docker network
create_network() {
    echo "Setting up Docker network..."
    docker network rm "$NETWORK_NAME" 2>/dev/null || true
    docker network create "$NETWORK_NAME"
}

# Function to deploy an LLM container
deploy_llm() {
    local name="$1"
    local port="$2"
    local gpu="$3"
    local project="$4"

    echo "Deploying $name on GPU $gpu, port $port..."
    docker run -d \
        --name "$name" \
        --network "$NETWORK_NAME" \
        -v "$SHARED_CACHE_DIR":/root/.cache/huggingface \
        --gpus "device=$gpu" \
        --shm-size=1g \
        -p "$port:$port" \
        -e CUDA_VISIBLE_DEVICES="$gpu" \
        -e PORT="$port" \
        -e RUN_MODE=server \
        -e MODEL_TYPE=LLM \
        -e SERVICE_NAME="$name" \
        -e LLM_RESPONSE_SCHEMA="EvidenceLLMResponse" \
        -e LANGCHAIN_PROJECT="$project" \
        -e LLM_MODEL_NAME="$MODEL_NAME" \
        -e HUGGING_FACE_HUB_TOKEN="$HF_TOKEN" \
        -e LANGCHAIN_API_KEY="$LANGCHAIN_API_KEY" \
        -e PEER_SERVICE_URL="http://${name}:${port}" \
        "$IMAGE_NAME"
}

# Function to deploy Streamlit UI
deploy_streamlit() {
    local port="$1"
    
    echo "Deploying Streamlit UI on port $port..."
    docker run -d \
        --name llm-ui \
        --network "$NETWORK_NAME" \
        -p "$port:$port" \
        -e RUN_MODE=ui \
        -e PORT="$port" \
        -e LLM1_URL="http://llm1:8888" \
        -e LLM2_URL="http://llm2:8889" \
        "$IMAGE_NAME"
}

# Function to check container health
check_container_health() {
    local name="$1"
    local max_attempts=30
    local attempt=1

    echo "Checking health of $name..."
    while [ $attempt -le $max_attempts ]; do
        if docker logs "$name" 2>&1 | grep -q "Application startup complete"; then
            echo "$name is healthy!"
            return 0
        fi
        echo "Waiting for $name to be healthy... (attempt $attempt/$max_attempts)"
        sleep 4
        ((attempt++))
    done
    
    echo "Error: $name failed to become healthy"
    return 1
}

# Function to display container logs
show_logs() {
    local name="$1"
    echo "Last few logs from $name:"
    docker logs "$name" --tail 20
}

# Main deployment process
main() {
    echo "Starting multi-LLM deployment..."
    
    # Initial checks
    check_docker
    check_nvidia_docker
    
    # Setup
    cleanup
    create_network
    
    # Deploy LLMs
    deploy_llm "llm1" "8888" "0" "demo-llm1"
    deploy_llm "llm2" "8889" "1" "demo-llm2"
    
    # Deploy Streamlit UI (optional)
    # deploy_streamlit "8890"
    
    # Check health
    for container in llm1 llm2; do
        if ! check_container_health "$container"; then
            echo "Deployment failed. Showing logs..."
            show_logs "$container"
            cleanup
            exit 1
        fi
    done
    
    echo "Deployment successful!"
    echo "LLM1 available at: http://localhost:8888"
    echo "LLM2 available at: http://localhost:8889"
    # echo "Streamlit UI available at: http://localhost:8890"
}

# Run the deployment
main

# Setup SSH tunneling (if running on remote machine)
setup_ssh_tunneling() {
    echo "To access the services locally, run these commands in separate terminal windows:"
    echo "ssh -L 8888:localhost:8888 <username>@<remote-host>"
    echo "ssh -L 8889:localhost:8889 <username>@<remote-host>"
    # echo "ssh -L 8890:localhost:8890 <username>@<remote-host>  # For Streamlit UI"
}

setup_ssh_tunneling

# Example usage:
echo ""
echo "Example API calls:"
echo 'curl -X POST "http://localhost:8888/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is 7+8?\"}"'
echo 'curl -X POST "http://localhost:8889/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is 7+8?\"}"'
