#!/bin/bash
set -e  # Exit on any error

# Function to display script usage
usage() {
    echo "Usage: $0 -h <hugging_face_token> -l <langchain_api_key>"
    echo "  -h    Hugging Face token"
    echo "  -l    LangChain API key"
    exit 1
}

# Parse command line arguments
while getopts "h:l:" opt; do
    case $opt in
        h) HF_TOKEN="$OPTARG" ;;
        l) LANGCHAIN_API_KEY="$OPTARG" ;;
        ?) usage ;;
    esac
done

# Validate required tokens
if [ -z "$HF_TOKEN" ] || [ -z "$LANGCHAIN_API_KEY" ]; then
    echo "Error: Hugging Face token and LangChain API key are required"
    usage
fi

# Function to make scripts executable
make_executable() {
    local script=$1
    if [ -f "$script" ]; then
        chmod +x "$script"
        echo "Made $script executable"
    else
        echo "Warning: $script not found"
        exit 1
    fi
}

echo "Starting setup process..."

# Make all scripts executable
make_executable "v2_deploy_multi_llm.sh"
make_executable "generate_env.sh"
make_executable "wait_for_service.sh"

# Generate .env file
./generate_env.sh "$HF_TOKEN" "$LANGCHAIN_API_KEY"

echo -e "\nSetup completed successfully!"
echo -e "\nYou can now run the deployment with:"
echo "./v2_deploy_multi_llm.sh -h \"$HF_TOKEN\" -l \"$LANGCHAIN_API_KEY\""
