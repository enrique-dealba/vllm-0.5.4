#!/bin/bash

# Default values
DEFAULT_DB_USER="postgres"
DEFAULT_DB_PASSWORD="password"
DEFAULT_DB_NAME="postgres"
DEFAULT_HF_TOKEN=""
DEFAULT_LANGCHAIN_TOKEN=""

export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:$LD_LIBRARY_PATH"

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -u, --user              Database user (default: postgres)"
    echo "  -p, --password          Database password (default: password)"
    echo "  -d, --database          Database name (default: postgres)"
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

echo "======================="
echo "STEP 1: Cleanup"
echo "======================="
echo "Cleaning up existing Docker resources..."
docker compose down -v --remove-orphans 2>/dev/null || true
docker rm -f timescaledb 2>/dev/null || true
docker volume rm timescaledb_data 2>/dev/null || true

echo "======================="
echo "STEP 2: Environment Setup"
echo "======================="
echo "Generating .env file..."
cat > .env << EOF
POSTGRES_DB=$DB_NAME
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASSWORD
TIMESCALE_SERVICE_URL=postgres://${DB_USER}:${DB_PASSWORD}@timescaledb:5432/${DB_NAME}
LLM_MODEL_NAME=mistralai/Mistral-Small-Instruct-2409
IS_MISTRAL=true
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
LANGCHAIN_API_KEY=$LANGCHAIN_TOKEN
LANGCHAIN_PROJECT=llm-project
LANGCHAIN_TRACING_V2=true
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib
EOF

echo "Contents of .env file:"
echo "======================="
cat .env
echo "======================="

chmod +x scripts/init-db.sh
chmod +x scripts/verify_*.sh

echo "======================="
echo "STEP 3: Database Setup"
echo "======================="
echo "Starting database service..."
docker compose up --build -d timescaledb
echo "Waiting for database initialization..."
sleep 10  # Give some time for init-db.sh to complete

echo "======================="
echo "STEP 4: Build"
echo "======================="
echo "Building application environment..."
docker compose build

echo "======================="
echo "STEP 5: Service Startup"
echo "======================="
echo "Starting services..."
docker compose up -d

# Check if containers are running
if [ $? -eq 0 ]; then
    echo "======================="
    echo "STEP 6: Status"
    echo "======================="
    echo "Services are now running!"
    echo "Database connection details:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  Database: $DB_NAME"
    echo "  User: $DB_USER"
    echo "  Password: $DB_PASSWORD"
    echo
    echo "Application is accessible at:"
    echo "  http://localhost:8888"
else
    echo "Error: Failed to start services"
    exit 1
fi
