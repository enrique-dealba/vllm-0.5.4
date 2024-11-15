#!/bin/bash

# Default values
DEFAULT_DB_USER="test_user"
DEFAULT_DB_PASSWORD="test_password"
DEFAULT_DB_NAME="test_db"
DEFAULT_HF_TOKEN=""
DEFAULT_LANGCHAIN_TOKEN=""

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -u, --user              Database user (default: test_user)"
    echo "  -p, --password          Database password (default: test_password)"
    echo "  -d, --database          Database name (default: test_db)"
    echo "  -h, --hf-token          Hugging Face token"
    echo "  -l, --langchain-token   LangChain token"
    echo "  --help                  Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--user)
            DB_USER="$2"
            shift 2
            ;;
        -p|--password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        -d|--database)
            DB_NAME="$2"
            shift 2
            ;;
        -h|--hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        -l|--langchain-token)
            LANGCHAIN_TOKEN="$2"
            shift 2
            ;;
        --help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

# Set default values if not provided
DB_USER=${DB_USER:-$DEFAULT_DB_USER}
DB_PASSWORD=${DB_PASSWORD:-$DEFAULT_DB_PASSWORD}
DB_NAME=${DB_NAME:-$DEFAULT_DB_NAME}
HF_TOKEN=${HF_TOKEN:-$DEFAULT_HF_TOKEN}
LANGCHAIN_TOKEN=${LANGCHAIN_TOKEN:-$DEFAULT_LANGCHAIN_TOKEN}

# Run cleanup first
echo "Cleaning up existing containers..."
docker compose -f docker-compose.test.yml down -v --remove-orphans

# Generate .env file
echo "Generating .env file..."
cat > .env << EOF
# Database Configuration
POSTGRES_DB=$DB_NAME
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASSWORD
TIMESCALE_SERVICE_URL=postgres://${DB_USER}:${DB_PASSWORD}@test_db:5432/${DB_NAME}

# Model Configuration
LLM_MODEL_NAME=mistralai/Mistral-Small-Instruct-2409
IS_MISTRAL=true

# API Tokens
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
LANGCHAIN_API_KEY=$LANGCHAIN_TOKEN
LANGCHAIN_PROJECT=test-postgres-project
LANGCHAIN_TRACING_V2=true

LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH
EOF

echo "Environment file created successfully!"

# Build and run tests
echo "Building test environment..."
docker compose -f docker-compose.test.yml build

echo "Verifying library setup..."
docker compose -f docker-compose.test.yml run tests /usr/local/bin/verify_libs.sh

if [ $? -ne 0 ]; then
    echo "Library verification failed!"
    exit 1
fi

# Before running tests, verify environment
echo "Verifying environment..."
docker compose -f docker-compose.test.yml run \
    -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH" \
    tests /bin/bash -c "./verify_env.sh && pytest tests/ -v --log-cli-level=INFO"

echo "Running tests..."
docker compose -f docker-compose.test.yml run tests

# Cleanup
docker compose -f docker-compose.test.yml down -v