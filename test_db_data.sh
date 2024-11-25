#!/bin/bash
set -e

# Default values
DEFAULT_HF_TOKEN=""
DEFAULT_LANGCHAIN_TOKEN=""
CONTAINER_NAME="vllm-054-timescaledb-1"
DB_USER="postgres"
DB_NAME="postgres"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
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
    echo "Error: Both HF_TOKEN and LANGCHAIN_TOKEN are required"
    usage
fi

# Function to check data
check_data() {
    local result=$(docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -t -c \
        "SELECT COUNT(*) FROM embeddings WHERE content = 'Test persistence';" 2>/dev/null)
    echo $result | tr -d ' '
}

# Function to run setup.sh quietly
run_setup() {
    echo "Running setup.sh..."
    ./setup.sh --hf-token "$HF_TOKEN" --langchain-token "$LANGCHAIN_TOKEN" > /dev/null 2>&1
}

# Main test sequence
echo "Starting database persistence test..."

# Step 1: Initial setup
echo "Step 1: Running initial setup..."
run_setup

# Step 2: Initial check
echo "Step 2: Checking initial state..."
initial_count=$(check_data)
echo "Initial record count: $initial_count"

# Step 3: Insert test data
echo "Step 3: Inserting test data..."
docker exec $CONTAINER_NAME psql -U $DB_USER -d $DB_NAME -c \
    "INSERT INTO embeddings (id, metadata, content, embedding) VALUES (gen_random_uuid(), '{\"source\": \"test\"}', 'Test persistence', array_fill(0.1, ARRAY[384]));" > /dev/null

# Step 4: Verify insertion
echo "Step 4: Verifying insertion..."
after_insert_count=$(check_data)
echo "Record count after insertion: $after_insert_count"

if [ "$after_insert_count" -ne "1" ]; then
    echo -e "${RED}ERROR: Data insertion failed${NC}"
    exit 1
fi

# Step 5: Rerun setup
echo "Step 5: Running setup again..."
run_setup

# Step 6: Final verification
echo "Step 6: Final verification..."
final_count=$(check_data)
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
