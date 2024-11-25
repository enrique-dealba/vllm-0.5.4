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
CLEANUP_TYPE="soft"
for arg in "$@"; do
    case $arg in
        --full-cleanup)
        CLEANUP_TYPE="full"
        shift
        ;;
        --cleanup-help)
        cleanup_usage
        exit 0
        ;;
    esac
done

echo "Performing $CLEANUP_TYPE cleanup..."

if [ "$CLEANUP_TYPE" = "full" ]; then
    echo "WARNING: Performing full cleanup including volumes..."
    docker compose -f docker-compose.test.yml down -v --remove-orphans
    docker volume rm test_timescaledb_data 2>/dev/null || true
else
    echo "Performing soft cleanup (preserving volumes)..."
    docker compose -f docker-compose.test.yml down --remove-orphans
fi

# Generate .env file
echo "======================="
echo "STEP 2: Environment Setup"
echo "======================="
echo "Generating .env file..."
cat > .env << EOF
POSTGRES_DB=$DB_NAME
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASSWORD
TIMESCALE_SERVICE_URL=postgres://${DB_USER}:${DB_PASSWORD}@test_db:5432/${DB_NAME}
LLM_MODEL_NAME=mistralai/Mistral-Small-Instruct-2409
IS_MISTRAL=true
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
LANGCHAIN_API_KEY=$LANGCHAIN_TOKEN
LANGCHAIN_PROJECT=test-postgres-project
LANGCHAIN_TRACING_V2=true
LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/lib
EOF
echo "Environment file created!"

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
docker compose -f docker-compose.test.yml up --build -d test_db
echo "Waiting for database initialization..."
sleep 10  # Give some time for init-db.sh to complete

echo "======================="
echo "STEP 4: Build"
echo "======================="
echo "Building test environment..."
docker compose -f docker-compose.test.yml build

echo "======================="
echo "STEP 5: Library Verification"
echo "======================="
echo "Running library verification..."

echo "Executing verification script..."
docker compose -f docker-compose.test.yml run verify

if [ $? -ne 0 ]; then
    echo "ERROR: Library verification failed!"
    exit 1
fi

echo "======================="
echo "STEP 6: Environment Verification"
echo "======================="
echo "Running environment verification..."
docker compose -f docker-compose.test.yml run \
    -e LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/lib:$LD_LIBRARY_PATH" \
    tests /bin/bash -c "/app/scripts/verify_env.sh"

echo "======================="
echo "STEP 7: Running Tests"
echo "======================="
echo "Executing pytest..."

# Use explicit test discovery
docker compose -f docker-compose.test.yml run \
    -e PYTHONPATH=/app \
    tests bash -c "
        source /root/miniconda3/bin/activate vllm && \
        cd /app && \
        python -m pytest \
            --verbose \
            --log-cli-level=INFO \
            --capture=no \
            -v \
            tests/
    "

TEST_EXIT_CODE=$?

if [ $TEST_EXIT_CODE -ne 0 ]; then
    echo "ERROR: Pytest execution failed with code $TEST_EXIT_CODE"
    exit 1
fi

echo "======================="
echo "STEP 8: Testing Volume Persistence"
echo "======================="
echo "Running volume persistence tests..."
chmod +x scripts/test_volume_persistence.sh
./scripts/test_volume_persistence.sh

if [ $? -ne 0 ]; then
    echo "ERROR: Volume persistence test failed!"
    exit 1
fi

echo "======================="
echo "STEP 9: Final Verification"
echo "======================="
echo "Verifying data persistence..."
docker exec vllm-054-timescaledb-1 psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT COUNT(*) FROM embeddings WHERE content LIKE 'Test persistence%';"

echo "======================="
echo "STEP 10: Final Cleanup"
echo "======================="
if [ "$CLEANUP_TYPE" = "full" ]; then
    echo "Performing full cleanup including volumes..."
    docker compose -f docker-compose.test.yml down -v --remove-orphans
else
    echo "Performing soft cleanup (preserving volumes)..."
    docker compose -f docker-compose.test.yml down --remove-orphans
fi