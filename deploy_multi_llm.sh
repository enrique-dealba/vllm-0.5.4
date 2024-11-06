#!/bin/bash
set -e  # Exit on any error

# Configuration
COMPOSE_FILE="docker-compose.yml"
HF_TOKEN="your-huggingface-token"  # Replace or pass as argument
LANGCHAIN_API_KEY="your-langchain-api-key"  # Replace or pass as argument

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

# Function to build Docker images
build_docker_images() {
    echo "Building Docker images..."
    docker compose -f "$COMPOSE_FILE" build
}

# Function to deploy services
deploy_services() {
    echo "Deploying services using docker compose..."
    docker compose -f "$COMPOSE_FILE" up -d
}

# Function to wait for services to become healthy
wait_for_services() {
    echo "Waiting for services to become healthy..."
    ./wait_for_service.sh
}

# Function to show GPU status
show_gpu_status() {
    echo -e "\nCurrent GPU Status:"
    nvidia-smi
}

# Function to cleanup existing deployments
cleanup() {
    echo "Cleaning up existing deployments..."
    
    # Kill any existing Python processes using GPUs
    nvidia-smi --query-compute-apps=pid --format=csv,noheader | while read -r pid; do
        if [ ! -z "$pid" ]; then
            echo "Killing process $pid using GPU..."
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    
    # Stop any running containers using the ports
    for port in 8000 8881 8882; do
        container_id=$(docker container ls -q --filter "publish=$port")
        if [ ! -z "$container_id" ]; then
            echo "Stopping container using port $port..."
            docker stop $container_id
        fi
    done

    # Remove containers from this deployment
    docker compose -f "$COMPOSE_FILE" down --remove-orphans

    echo "Cleanup completed"
    
    # Wait for GPU memory to clear
    sleep 5
}

# Main deployment process
main() {
    echo "Starting multi-LLM deployment using docker-compose..."

    # Initial checks
    check_docker

    # Cleanup before deployment
    cleanup

    # Export tokens as environment variables for docker-compose
    export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
    export LANGCHAIN_API_KEY="$LANGCHAIN_API_KEY"

    # Build Docker images
    build_docker_images

    # Deploy services
    deploy_services

    # Wait for services to become healthy
    wait_for_services

    echo "Deployment successful!"
    echo "Main Server available at: http://localhost:8000"
    echo "LLM1 available at: http://localhost:8881"
    echo "LLM2 available at: http://localhost:8882"

    # Show GPU status
    show_gpu_status
}

# Run the deployment
main

# Example API calls
echo -e "\nExample API calls:"
echo 'curl -X POST "http://localhost:8000/meta/generate" -H "Content-Type: application/json" -d "{\"text\": \"What is 7+8?\"}"'

# Health check example
echo -e "\nHealth check endpoints:"
echo 'curl http://localhost:8000/health'
echo 'curl http://localhost:8881/health'
echo 'curl http://localhost:8882/health'
