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
    echo "  --full                  Perform a full cleanup including volumes"
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
        --full)
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

if [ "$CLEANUP_TYPE" = "full" ]; then
    echo "WARNING: Performing full cleanup including volumes..."
    read -p "This will delete all data. Are you sure? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        docker compose down --remove-orphans
        sudo rm -rf ../db_data
        echo "db_data directory removed"
        mkdir ../db_data
        sudo chown -R 1000:1000 ../db_data
        sudo chmod -R 700 ../db_data
    else
        echo "Aborted full cleanup"
        exit 1
    fi
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
    echo "Cleanup complete. You may need to check volume permissions manually."
    exit 1
}

# Set error trap
trap cleanup_on_error ERR

echo "======================="
echo "STEP 3: Volume Check"
echo "======================="
echo "Checking Docker bind mount directory..."

# Function to safely check directory
check_directory() {
    local dir="$1"
    if sudo test -d "$dir"; then
        echo "$dir directory exists, checking contents..."
        if sudo test "$(sudo ls -A $dir)"; then
            echo "$dir directory contains data, preserving..."
        else
            echo "$dir directory is empty, preparing for initialization..."
        fi
    else
        echo "Creating $dir directory"
        sudo mkdir -p "$dir"
    fi
}

# Function to set permissions
set_permissions() {
    local dir="$1"
    echo "Setting correct permissions for $dir..."
    sudo chown -R 1000:1000 "$dir"
    sudo chmod -R 700 "$dir"
    
    # Verify permissions
    if sudo test -w "$dir"; then
        echo "Permissions set successfully"
    else
        echo "ERROR: Failed to set proper permissions on $dir"
        exit 1
    fi
}

# Main execution
DB_DATA_DIR="../db_data"

# Check and create directory if needed
check_directory "$DB_DATA_DIR"

# Set proper permissions
set_permissions "$DB_DATA_DIR"

# Final verification
if ! sudo test -w "$DB_DATA_DIR"; then
    echo "ERROR: $DB_DATA_DIR directory is not writable"
    echo "Current permissions:"
    ls -la "$DB_DATA_DIR"
    echo "Current owner:"
    stat -c "%U:%G" "$DB_DATA_DIR"
    exit 1
fi

echo "$DB_DATA_DIR directory prepared successfully"

echo "======================="
echo "STEP 4: Database Setup"
echo "======================="
echo "Starting database service..."
docker compose up --build -d timescaledb

# Wait for container and verify initialization
max_attempts=15
attempt=1
while [ $attempt -le $max_attempts ]; do
    # Get container ID using docker compose
    container_id=$(docker compose ps -q timescaledb)

    if [ -n "$container_id" ]; then
        health_status=$(docker inspect -f '{{.State.Health.Status}}' "$container_id" 2>/dev/null || echo "unhealthy")
        
        if [ "$health_status" = "healthy" ]; then
            # Verify table exists and is accessible
            if docker exec "$container_id" psql -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT COUNT(*) FROM embeddings;" >/dev/null 2>&1; then
                echo "Database initialization verified successfully!"
                break
            fi
        fi
        
        echo "Waiting for database... Attempt $attempt/$max_attempts"
        echo "Container Health: $health_status"
        
        # Show logs if halfway through attempts
        if [ $attempt -eq $((max_attempts / 2)) ]; then
            echo "Database logs:"
            docker logs "$container_id"
        fi
    else
        echo "Container not found. Attempt $attempt/$max_attempts"
    fi
    
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
echo "STEP 8: Data Verification"
echo "======================="
echo "Verifying data persistence..."
container_id=$(docker compose ps -q timescaledb)
docker exec "$container_id" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT COUNT(*) FROM embeddings;"

echo "======================="
echo "Setup Complete"
echo "======================="