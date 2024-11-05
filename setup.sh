#!/bin/bash
set -e  # Exit on any error

# Function to display script usage
usage() {
   echo "Usage: $0 -h <hugging_face_token> -l <langchain_api_key> [options]"
   echo "  -h    Hugging Face token (required)"
   echo "  -l    LangChain API key (required)"
   echo "  -m    LLM1 model name (default: mistralai/Mistral-Small-Instruct-2409)"
   echo "  -t    LLM1 model type (LLM or VLM, default: LLM)"
   echo "  -p    LLM1 temperature (default: 0.2)"
   echo "  -x    LLM1 max tokens (default: 8192)"
   exit 1
}

# Parse command line arguments
while getopts "h:l:m:t:p:x:" opt; do
   case $opt in
       h) HF_TOKEN="$OPTARG" ;;
       l) LANGCHAIN_API_KEY="$OPTARG" ;;
       m) LLM_MODEL_NAME="$OPTARG" ;;
       t) MODEL_TYPE="$OPTARG" ;;
       p) TEMPERATURE="$OPTARG" ;;
       x) MAX_TOKENS="$OPTARG" ;;
       ?) usage ;;
   esac
done

# Validate required tokens
if [ -z "$HF_TOKEN" ] || [ -z "$LANGCHAIN_API_KEY" ]; then
   echo "Error: Hugging Face token and LangChain API key are required"
   usage
fi

# Function to check disk space
check_disk_space() {
   local required_space=50  # GB
   local docker_root=$(docker info --format '{{.DockerRootDir}}')
   local available_space=$(df -BG "$docker_root" | awk 'NR==2 {print $4}' | sed 's/G//')
   
   if [ "$available_space" -lt "$required_space" ]; then
       echo "Error: Not enough disk space in Docker root directory. Available: ${available_space}GB, Required: ${required_space}GB"
       exit 1
   fi
   echo "Docker storage space check passed. Available: ${available_space}GB"
}

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

# Check disk space
check_disk_space

# Make all scripts executable
make_executable "deploy_multi_llm.sh"
make_executable "generate_env.sh"
make_executable "wait_for_service.sh"

# Export LLM1 configs for generate_env.sh
export LLM_MODEL_NAME=${LLM_MODEL_NAME}
export MODEL_TYPE=${MODEL_TYPE}
export TEMPERATURE=${TEMPERATURE}
export MAX_TOKENS=${MAX_TOKENS}

# Generate .env file
./generate_env.sh "$HF_TOKEN" "$LANGCHAIN_API_KEY"

echo -e "\nSetup completed successfully!"
echo -e "\nYou can now run the deployment with:"
echo "./deploy_multi_llm.sh -h \"$HF_TOKEN\" -l \"$LANGCHAIN_API_KEY\""
