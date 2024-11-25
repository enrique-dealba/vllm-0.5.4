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

# Function to check if container is ready
wait_for_container() {
    local start_time=$(date +%s)
    local timeout_time=$((start_time + TIMEOUT))
    
    while true; do
        if [ $(date +%s) -gt $timeout_time ]; then
            echo -e "${RED}Timeout waiting for container${NC}"
            return 1
        fi
        
        if docker ps | grep -q $CONTAINER_NAME; then
            local health=$(docker inspect -f '{{.State.Health.Status}}' $CONTAINER_NAME 2>/dev/null)
            if [ "$health" = "healthy" ]; then
                echo -e "${GREEN}Container is healthy${NC}"
                return 0
            fi
        fi
        
        echo -e "${YELLOW}Waiting for container to be ready...${NC}"
        sleep $RETRY_INTERVAL
    done
}

# Function to check data
check_data() {
    local result
    result=$(docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -t -c \
        "SELECT COUNT(*) FROM embeddings WHERE content = 'Test persistence';" 2>/dev/null || echo "ERROR")
    if [ "$result" = "ERROR" ]; then
        echo -e "${RED}Error executing database query${NC}"
        return 1
    fi
    echo $result | tr -d ' '
}

# Function to run setup.sh with timeout
run_setup() {
    echo "Running setup.sh..."
    timeout $TIMEOUT ./setup.sh --hf-token "$HF_TOKEN" --langchain-token "$LANGCHAIN_TOKEN" > /dev/null 2>&1
    if [ $? -eq 124 ]; then
        echo -e "${RED}Setup timed out after ${TIMEOUT} seconds${NC}"
        return 1
    fi
    echo -e "${GREEN}Setup completed${NC}"
    
    # Wait for container to be ready
    wait_for_container
}

# Main test sequence with error handling
echo "Starting database persistence test..."

# Step 1: Initial setup
echo "Step 1: Running initial setup..."
if ! run_setup; then
    echo -e "${RED}Initial setup failed${NC}"
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
if ! docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c \
    "INSERT INTO embeddings (id, metadata, content, embedding) VALUES (gen_random_uuid(), '{\"source\": \"test\"}', 'Test persistence', array_fill(0.1, ARRAY[384]));" > /dev/null 2>&1; then
    echo -e "${RED}Failed to insert test data${NC}"
    exit 1
fi

# Step 4: Verify insertion
echo "Step 4: Verifying insertion..."
after_insert_count=$(check_data)
if [ $? -ne 0 ] || [ "$after_insert_count" -ne "1" ]; then
    echo -e "${RED}Data insertion verification failed${NC}"
    exit 1
fi
echo "Record count after insertion: $after_insert_count"

# Step 5: Rerun setup
echo "Step 5: Running setup again..."
if ! run_setup; then
    echo -e "${RED}Second setup run failed${NC}"
    exit 1
fi

# Step 6: Final verification
echo "Step 6: Final verification..."
final_count=$(check_data)
if [ $? -ne 0 ]; then
    echo -e "${RED}Final verification failed${NC}"
    exit 1
fi
echo "Final record count: $final_count"

# Check results
if [ "$final_count" -eq "1" ]; then
    echo -e "${GREEN}SUCCESS: Data persisted after setup rerun${NC}"
    echo "Test record details:"
    docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c \
        "SELECT id, metadata, content, created_at FROM embeddings WHERE content = 'Test persistence';"
    exit 0
else
    echo -e "${RED}FAILURE: Data did not persist after setup rerun${NC}"
    echo "Expected 1 record, found $final_count"
    exit 1
fi
