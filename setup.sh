#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Initialize variables
HF_TOKEN=""
LANGCHAIN_KEY=""

# Function to display usage
usage() {
   echo "Usage: $0 -h <Hugging Face token> -l <LangChain API key>"
   echo "  -h    Hugging Face token"
   echo "  -l    LangChain API key"
   exit 1
}

# Parse command line arguments
while getopts "h:l:" opt; do
   case $opt in
       h) HF_TOKEN="$OPTARG" ;;
       l) LANGCHAIN_KEY="$OPTARG" ;;
       *) usage ;;
   esac
done

# Check if both arguments are provided
if [ -z "$HF_TOKEN" ] || [ -z "$LANGCHAIN_KEY" ]; then
   usage
fi

# Create or overwrite .env file
echo "Generating .env file..."
cat > .env << EOF
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
LANGCHAIN_API_KEY=$LANGCHAIN_KEY
EOF

echo ".env file created successfully!"
echo "Contents of .env:"
cat .env

# Validate docker-compose.yml before proceeding
echo "Validating docker-compose.yml..."
if ! docker compose config > /dev/null 2>&1; then
    echo "Error: docker-compose.yml is invalid. Please check the file for syntax errors."
    exit 1
fi

# Bring down any existing containers to ensure a clean start
echo "Stopping and removing any existing Docker containers..."
docker compose down

# Start the Docker containers in detached mode
echo "Starting Docker containers..."
docker compose up -d

# Provide feedback to the user
echo "Docker Compose services are up and running."
echo "Waiting for Milvus to become healthy..."

# Wait until Milvus is healthy or timeout after 120 seconds
TIMEOUT=120
INTERVAL=5
elapsed=0

while ! curl -sSf http://localhost:19121/health >/dev/null 2>&1; do
    if [ $elapsed -ge $TIMEOUT ]; then
        echo "Error: Milvus did not become healthy within $TIMEOUT seconds."
        echo "Check the logs using 'docker compose logs milvus' for more information."
        exit 1
    fi
    echo "Milvus is not healthy yet. Waiting..."
    sleep $INTERVAL
    elapsed=$((elapsed + INTERVAL))
done

echo "Milvus is healthy and running successfully!"
