#!/bin/bash
set -e

# Default values and constants
DEFAULT_HF_TOKEN=""
DEFAULT_LANGCHAIN_TOKEN=""
CONTAINER_NAME="vllm-054-timescaledb-1"
DB_USER="postgres"
DB_NAME="postgres"
TIMEOUT=300  # 5 minutes timeout
RETRY_INTERVAL=5

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Debug mode
DEBUG=true

chmod +x setup.sh

# Debug function
debug() {
    if [ "$DEBUG" = true ]; then
        echo -e "${YELLOW}DEBUG: $1${NC}"
    fi
}

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -h, --hf-token          Hugging Face token (required)"
    echo "  -l, --langchain-token   LangChain token (required)"
    echo "  --help                  Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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

# Verify required parameters
if [ -z "$HF_TOKEN" ] || [ -z "$LANGCHAIN_TOKEN" ]; then
    echo -e "${RED}Error: Both HF_TOKEN and LANGCHAIN_TOKEN are required${NC}"
    usage
fi

wait_for_container() {
    local container_id=$1
    local max_attempts=10
    local wait_time=3
    
    debug "Waiting for container $container_id to be ready..."
    
    for ((i=1; i<=max_attempts; i++)); do
        debug "Checking container status (attempt $i/$max_attempts)"
        
        # Check if container exists and is running
        local status=$(docker inspect -f '{{.State.Status}}' "$container_id" 2>/dev/null)
        local health=$(docker inspect -f '{{.State.Health.Status}}' "$container_id" 2>/dev/null)
        
        debug "Status: '$status', Health: '$health'"
        
        if [ "$status" = "running" ]; then
            if [ "$health" = "healthy" ] || [ "$health" = "<nil>" ]; then
                debug "Container is running and healthy"
                return 0
            fi
        fi
        
        debug "Waiting ${wait_time}s before next check..."
        sleep $wait_time
    done
    
    echo -e "${RED}Container failed to reach running state within timeout${NC}"
    return 1
}

get_container_id() {
    local retries=5
    local wait_time=2
    local container_id=""
    
    for ((i=1; i<=retries; i++)); do
        debug "Attempting to get container ID (attempt $i/$retries)"
        container_id=$(docker ps -qf "name=$CONTAINER_NAME")
        
        if [ ! -z "$container_id" ]; then
            debug "Container ID found: $container_id"
            echo "$container_id"
            return 0
        fi
        
        debug "Container not found, waiting ${wait_time}s..."
        sleep $wait_time
    done
    
    return 1
}

# Function to check data
check_data() {
    debug "Checking data in database..."
    local result
    local container_id
    
    container_id=$(get_container_id)
    if [ -z "$container_id" ]; then
        echo -e "${RED}Error: Could not find container${NC}"
        return 1
    fi
    
    debug "Using container ID: $container_id"
    
    result=$(docker exec "$container_id" psql -U "$DB_USER" -d "$DB_NAME" -t -c \
        "SELECT COUNT(*) FROM embeddings WHERE content = 'Test persistence';" 2>/dev/null || echo "ERROR")
    
    if [ "$result" = "ERROR" ]; then
        echo -e "${RED}Error executing database query${NC}"
        docker logs "$container_id" | tail -n 20
        return 1
    fi
    
    debug "Query result: '$result'"
    echo "$result" | tr -d ' \n'
}

# Function to run setup.sh with debugging
run_setup() {
    local setup_output_file=$(mktemp)
    echo "Running setup.sh..."
    debug "Running setup.sh with tokens..."
    
    if ! ./setup.sh --hf-token "$HF_TOKEN" --langchain-token "$LANGCHAIN_TOKEN" > "$setup_output_file" 2>&1; then
        echo -e "${RED}Setup failed. Output:${NC}"
        cat "$setup_output_file"
        rm "$setup_output_file"
        return 1
    fi
    
    debug "Setup.sh output:"
    cat "$setup_output_file"
    rm "$setup_output_file"
    
    debug "Waiting for container initialization..."
    sleep 5  # Initial wait for container creation
    
    # Get container ID
    local container_id
    container_id=$(get_container_id)
    if [ -z "$container_id" ]; then
        echo -e "${RED}Container $CONTAINER_NAME not found after setup${NC}"
        return 1
    fi
    
    # Wait for container to be ready
    if ! wait_for_container "$container_id"; then
        echo -e "${RED}Container failed to start properly${NC}"
        docker logs "$container_id"
        return 1
    fi
    
    debug "Container is ready"
    debug "Container details:"
    docker ps --filter "id=$container_id" --format "table {{.ID}}\t{{.Status}}\t{{.Names}}"
    
    # Verify database is accepting connections
    debug "Verifying database connection..."
    local max_db_attempts=5
    for ((i=1; i<=max_db_attempts; i++)); do
        if docker exec "$container_id" pg_isready -U "$DB_USER" >/dev/null 2>&1; then
            debug "Database is accepting connections"
            break
        fi
        if [ $i -eq $max_db_attempts ]; then
            echo -e "${RED}Database failed to accept connections${NC}"
            return 1
        fi
        debug "Waiting for database to accept connections (attempt $i/$max_db_attempts)..."
        sleep 3
    done
    
    debug "Container logs:"
    docker logs "$container_id" | tail -n 20
    
    return 0
}

check_volume_persistence() {
    debug "Checking volume persistence..."

    PGDATA_PATH="/home/edealba/Testing/TestingLLMs/postgres-vllm/db_data"

    if [ ! -d "$PGDATA_PATH" ]; then
        echo -e "${RED}Error: db_data directory not found at $PGDATA_PATH${NC}"
        return 1
    fi

    if [ ! -f "$PGDATA_PATH/PG_VERSION" ]; then
        echo -e "${RED}Error: Database files not properly persisted (PG_VERSION missing)${NC}"
        return 1
    fi

    if [ ! -f "$PGDATA_PATH/.initialized" ]; then
        echo -e "${RED}Error: Database not properly initialized (.initialized marker missing)${NC}"
        return 1
    fi

    debug "Volume appears to be properly persisted at $PGDATA_PATH"
    return 0
}

# Main test sequence with debugging
echo "Starting database persistence test..."

# Step 1: Initial setup
echo "Step 1: Running initial setup..."
if ! run_setup; then
    echo -e "${RED}Initial setup failed${NC}"
    exit 1
fi

if ! check_volume_persistence; then
    echo -e "${RED}Volume persistence check failed${NC}"
    exit 1
fi

# Step 2: Initial check
echo "Step 2: Checking initial state..."
initial_count=$(check_data)
if [ $? -ne 0 ]; then
    echo -e "${RED}Failed to check initial state${NC}"
    exit 1
fi
echo "Initial record count: $initial_count"

# Step 3: Insert test data
echo "Step 3: Inserting test data..."
debug "Executing INSERT query..."
if ! docker exec "$CONTAINER_ID" psql -U "$DB_USER" -d "$DB_NAME" -c \
    "INSERT INTO embeddings (id, metadata, content, embedding) VALUES (gen_random_uuid(), '{\"source\": \"test\"}', 'Test persistence', array_fill(0.1, ARRAY[384]));" > /dev/null 2>&1; then
    echo -e "${RED}Failed to insert test data${NC}"
    exit 1
fi

# Step 4: Verifying insertion
echo "Step 4: Verifying insertion..."
after_insert_count=$(check_data)
if [ $? -ne 0 ] || [ "$after_insert_count" -ne "1" ]; then
    echo -e "${RED}Data insertion verification failed${NC}"
    exit 1
fi
echo "Record count after insertion: $after_insert_count"

# Step 5: Rerun setup
echo "Step 5: Running setup again..."
debug "Running setup.sh second time..."
if ! run_setup; then
    echo -e "${RED}Second setup run failed${NC}"
    exit 1
fi

if ! check_volume_persistence; then
    echo -e "${RED}Volume persistence check failed${NC}"
    exit 1
fi

# Step 6: Final verification
echo "Step 6: Final verification..."
final_count=$(check_data)
debug "Final count: $final_count"
if [ "$final_count" = "ERROR" ]; then
    echo -e "${RED}Final verification failed${NC}"
    exit 1
fi

# Check results (use explicit -eq for integer comparison)
if [ "$final_count" -eq "1" ]; then
    echo -e "${GREEN}SUCCESS: Data persisted after setup rerun${NC}"
    echo "Test record details:"
    docker exec "$CONTAINER_ID" psql -U "$DB_USER" -d "$DB_NAME" -c \
        "SELECT id, metadata, content, created_at FROM embeddings WHERE content = 'Test persistence';"
else
    echo -e "${RED}FAILURE: Data did not persist after setup rerun${NC}"
    echo "Expected 1 record, found $final_count"
    debug "Current database state:"
    docker exec "$CONTAINER_ID" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT COUNT(*) FROM embeddings;"
    docker exec "$CONTAINER_ID" psql -U "$DB_USER" -d "$DB_NAME" -c "SELECT * FROM embeddings LIMIT 5;"
fi
