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
    echo "WARNING: Performing full cleanup including containers and networks, but preserving bind-mounted data..."
    read -p "This will stop and remove containers and networks. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down --remove-orphans
        # Do not remove volumes since we're using bind mounts
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

# Source the .env file to export variables
set -a
source .env
set +a

echo "Contents of .env file:"
echo "======================="
cat .env
echo "======================="

# Step 3: Database Setup
echo "======================="
echo "STEP 3: Database Setup"
echo "======================="
echo "Starting database service..."
docker compose up -d timescaledb

# Wait for the database to become healthy
attempt=1
max_attempts=15
while [ $attempt -le $max_attempts ]; do
    CONTAINER_ID=$(docker compose ps -q timescaledb)
    health_status=$(docker inspect --format='{{json .State.Health.Status}}' "$CONTAINER_ID" 2>/dev/null || echo "unhealthy")
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

# Step 4: Volume Check
echo "======================="
echo "STEP 4: Volume Check"
echo "======================="
echo "Checking bind-mounted data directory..."

PGDATA_PATH="/home/edealba/Testing/TestingLLMs/postgres-vllm/db_data"

if [ ! -d "$PGDATA_PATH" ]; then
    echo -e "${RED}Error: db_data directory not found at $PGDATA_PATH${NC}"
    exit 1
fi

echo "DEBUG: Current user info:"
whoami
echo "DEBUG: Current UID/GID:"
id
echo "DEBUG: Current groups:"
groups
echo "DEBUG: Parent directory permissions:"
ls -ld "$(dirname $PGDATA_PATH)"
echo "DEBUG: Target directory permissions:"
ls -ld "$PGDATA_PATH"
echo "DEBUG: Trying with sudo:"
sudo ls -la "$PGDATA_PATH"
echo "DEBUG: File ownership of PGDATA:"
stat -c "%U:%G" "$PGDATA_PATH"
echo "DEBUG: Full stat of PGDATA:"
stat "$PGDATA_PATH"

echo "DEBUG: Contents of $PGDATA_PATH:"
ls -la "$PGDATA_PATH"
echo "DEBUG: Looking for PG_VERSION at: $PGDATA_PATH/PG_VERSION"

echo "DEBUG: Testing direct file access:"
if [ -r "$PGDATA_PATH/PG_VERSION" ]; then
    echo "PG_VERSION is readable"
else
    echo "PG_VERSION is not readable"
fi
echo "DEBUG: Effective permissions:"
namei -l "$PGDATA_PATH/PG_VERSION"

if [ ! -f "$PGDATA_PATH/PG_VERSION" ]; then
    echo -e "${RED}Error: Database files not properly persisted (PG_VERSION missing)${NC}"
    exit 1
fi

if [ ! -f "$PGDATA_PATH/.initialized" ]; then
    echo -e "${RED}Error: Database not properly initialized (.initialized marker missing)${NC}"
    exit 1
fi

echo "Bind-mounted data directory is properly set up."
echo "======================="

# Step 5: Build
echo "======================="
echo "STEP 5: Build"
echo "======================="
echo "Building application environment..."
docker compose build

# Step 6: Service Startup
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