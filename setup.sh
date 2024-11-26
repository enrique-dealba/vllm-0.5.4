#!/bin/bash
set -e

# Default values and constants
DEFAULT_HF_TOKEN=""
DEFAULT_LANGCHAIN_TOKEN=""
CLEANUP_TYPE="soft"  # Default to soft cleanup

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)
            CLEANUP_TYPE="full"
            shift
            ;;
        --hf-token)
            HF_TOKEN="$2"
            shift 2
            ;;
        --langchain-token)
            LANGCHAIN_TOKEN="$2"
            shift 2
            ;;
        --help)
            echo "Usage: $0 [--full] [--hf-token TOKEN] [--langchain-token TOKEN]"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 [--full] [--hf-token TOKEN] [--langchain-token TOKEN]"
            exit 1
            ;;
    esac
done

# Step 1: Cleanup
echo "======================="
echo "STEP 1: Cleanup"
echo "======================="
if [ "$CLEANUP_TYPE" = "full" ]; then
    echo "WARNING: Performing full cleanup including volumes..."
    read -p "This will delete all data. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down --remove-orphans
        # No need to remove named volumes as we're using bind mounts
        echo "Full cleanup completed without removing bind-mounted data."
    else
        echo "Aborted full cleanup"
        exit 1
    fi
else
    echo "Performing soft cleanup (preserving bind-mounted data)..."
    docker compose stop
fi

# Step 2: Environment Setup
echo "======================="
echo "STEP 2: Environment Setup"
echo "======================="
echo "Generating .env file..."
cat <<EOF > .env
POSTGRES_DB=${POSTGRES_DB:-postgres}
POSTGRES_USER=${POSTGRES_USER:-postgres}
POSTGRES_PASSWORD=${POSTGRES_PASSWORD:-password}
TIMESCALE_SERVICE_URL=postgres://postgres:password@timescaledb:5432/postgres
LLM_MODEL_NAME=mistralai/Mistral-Small-Instruct-2409
IS_MISTRAL=true
HUGGING_FACE_HUB_TOKEN=${HF_TOKEN:-}
LANGCHAIN_API_KEY=${LANGCHAIN_TOKEN:-}
LANGCHAIN_PROJECT=llm-project
LANGCHAIN_TRACING_V2=true
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib
EOF

echo "Contents of .env file:"
echo "======================="
cat .env
echo "======================="

# Step 3: Volume Check
echo "======================="
echo "STEP 3: Volume Check"
echo "======================="
echo "Checking Docker named volume..."
# Docker named volumes are handled by Docker; no host directory permissions to set
echo "timescaledb_data volume is set to be managed by Docker."
echo "======================="

# Step 4: Database Setup
echo "======================="
echo "STEP 4: Database Setup"
echo "======================="
echo "Starting database service..."
docker compose up -d timescaledb

# Wait for the database to become healthy
attempt=1
max_attempts=15
while [ $attempt -le $max_attempts ]; do
    health_status=$(docker inspect --format='{{json .State.Health.Status}}' vllm-054-timescaledb-1 2>/dev/null || echo "unhealthy")
    if [ "$health_status" = "\"healthy\"" ]; then
        echo "Database is healthy."
        break
    fi
    echo "Waiting for database... Attempt $attempt/$max_attempts"
    echo "Container Health: $health_status"
    sleep 5
    attempt=$((attempt + 1))
done

if [ $attempt -gt $max_attempts ]; then
    echo "ERROR: Database initialization failed"
    docker ps -a  # List all containers to find the failed one
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
    echo "  Database: $POSTGRES_DB"
    echo "  User: $POSTGRES_USER"
    echo "  Password: $POSTGRES_PASSWORD"
    echo
    echo "Application is accessible at:"
    echo "  http://localhost:8888"
else
    echo "Error: Failed to start services"
    exit 1
fi

chmod +x scripts/test_db.sh

echo "======================="
echo "STEP 7: Database Initialization Verification"
echo "======================="
# Run the test_db_data.sh script's initialization verification
./scripts/test_db.sh  # Adjust the script name/path as necessary

echo "Database verification completed successfully."

echo "======================="
echo "STEP 8: Data Verification"
echo "======================="
echo "Verifying data persistence..."
container_id=$(docker compose ps -q timescaledb)
docker exec "$container_id" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT COUNT(*) FROM embeddings;"

echo "======================="
echo "Setup Complete"
echo "======================="