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
        --full|--full-cleanup)
            CLEANUP_TYPE="full"
            shift
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

# Function to show cleanup usage
cleanup_usage() {
    echo "Cleanup options:"
    echo "  --full     : Complete cleanup including volumes (WARNING: Deletes all data)"
    echo "  --soft     : Stop containers but preserve volumes (Default)"
}

# Parse cleanup type from arguments
# CLEANUP_TYPE="soft"
# for arg in "$@"; do
#     case $arg in
#         --full-cleanup)
#         CLEANUP_TYPE="full"
#         shift
#         ;;
#         --cleanup-help)
#         cleanup_usage
#         exit 0
#         ;;
#     esac
# done

echo "Performing $CLEANUP_TYPE cleanup..."

if [ "$CLEANUP_TYPE" = "full" ]; then
    echo "WARNING: Performing full cleanup including volumes..."
    docker compose down --volumes --remove-orphans
    docker volume rm timescaledb_data 2>/dev/null || true
    echo "Recreating volume..."
    docker volume create \
        --driver local \
        --label app=timescaledb \
        --label environment=production \
        timescaledb_data
else
    echo "Performing soft cleanup (preserving volumes)..."
    docker compose down --remove-orphans
fi

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
chmod +x scripts/test_db.sh
chmod +x scripts/verify_*.sh

set -e  # Exit on error

# Function for cleanup on error
cleanup_on_error() {
    echo "Error occurred. Cleaning up..."
    docker compose down 2>/dev/null || true
    # Don't remove volumes on error, just stop containers
    echo "Cleanup complete. You may need to check volume permissions manually."
    exit 1
}

# Set error trap
trap cleanup_on_error ERR

echo "======================="
echo "STEP 3: Volume Check"
echo "======================="
echo "Checking Docker volume..."
if docker volume inspect timescaledb_data >/dev/null 2>&1; then
    echo "TimescaleDB volume exists, preserving data"
else
    echo "Creating new TimescaleDB volume"
    # Create volume with specific driver and options
    docker volume create \
        --driver local \
        --label app=timescaledb \
        --label environment=production \
        timescaledb_data
fi

# Verify volume creation
if ! docker volume inspect timescaledb_data >/dev/null 2>&1; then
    echo "ERROR: Failed to create or verify TimescaleDB volume"
    exit 1
fi
echo "TimescaleDB volume verified"

# Let the Docker entrypoint handle permissions
echo "Initializing volume permissions..."

echo "======================="
echo "STEP 4: Database Setup"
echo "======================="
echo "Starting database service..."
docker compose up --build -d timescaledb

# Wait for container to be healthy
echo "Waiting for database to be ready..."
max_attempts=30
attempt=1
while [ $attempt -le $max_attempts ]; do
    container_id=$(docker ps --filter "name=vllm-054-timescaledb-1" --format "{{.ID}}")
    if [ -n "$container_id" ]; then
        if docker exec "$container_id" pg_isready -q; then
            echo "Database is ready!"
            
            # Verify table existence
            if docker exec "$container_id" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1 FROM embeddings LIMIT 1" >/dev/null 2>&1; then
                echo "Database initialization verified!"
                break
            fi
            
            echo "Running initialization..."
            docker exec "$container_id" bash /docker-entrypoint-initdb.d/init-db.sh
            
            if docker exec "$container_id" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1 FROM embeddings LIMIT 1" >/dev/null 2>&1; then
                echo "Database initialization completed successfully!"
                break
            else
                echo "ERROR: Database initialization failed"
                exit 1
            fi
        fi
    fi
    echo "Waiting for database... Attempt $attempt/$max_attempts"
    sleep 2
    attempt=$((attempt + 1))
done

if [ $attempt -gt $max_attempts ]; then
    echo "ERROR: Database failed to initialize after $max_attempts attempts"
    exit 1
fi

echo "======================="
echo "STEP 5: Build"
echo "======================="
echo "Building application environment..."
docker compose build

echo "======================="
echo "STEP 6: Service Startup"
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

echo "======================="
echo "STEP 7: Database Initialization Verification"
echo "======================="
# Run the test_db.sh script
./scripts/test_db.sh

echo "Database verification completed successfully."

echo "======================="
echo "Setup Complete"
echo "======================="