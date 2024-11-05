#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <huggingface_token> <langchain_api_key>"
    exit 1
fi

# Read defaults from model_config.yml
if [ -f "model_config.yml" ]; then
    # LLM1 defaults
    LLM1_MODEL_NAME_DEFAULT=$(yq eval '.llm1.model_name' model_config.yml)
    LLM1_MODEL_TYPE_DEFAULT=$(yq eval '.llm1.model_type' model_config.yml)
    LLM1_TEMPERATURE_DEFAULT=$(yq eval '.llm1.temperature' model_config.yml)
    LLM1_MAX_TOKENS_DEFAULT=$(yq eval '.llm1.max_tokens' model_config.yml)
    LLM1_IS_MISTRAL_DEFAULT=$(yq eval '.llm1.is_mistral' model_config.yml)

    # LLM2 defaults
    LLM2_MODEL_NAME_DEFAULT=$(yq eval '.llm2.model_name' model_config.yml)
    LLM2_MODEL_TYPE_DEFAULT=$(yq eval '.llm2.model_type' model_config.yml)
    LLM2_TEMPERATURE_DEFAULT=$(yq eval '.llm2.temperature' model_config.yml)
    LLM2_MAX_TOKENS_DEFAULT=$(yq eval '.llm2.max_tokens' model_config.yml)
    LLM2_IS_MISTRAL_DEFAULT=$(yq eval '.llm2.is_mistral' model_config.yml)
else
    echo "Warning: model_config.yml not found, using hardcoded defaults"
fi

# Create .env file with the provided arguments and model configs
cat > .env << EOF
# .env

# API Tokens
HUGGING_FACE_HUB_TOKEN=$1
LANGCHAIN_API_KEY=$2

# LLM1 Configuration (Command line overrides or defaults from model_config.yml)
LLM1_MODEL_NAME=${LLM_MODEL_NAME:-${LLM1_MODEL_NAME_DEFAULT}}
LLM1_MODEL_TYPE=${MODEL_TYPE:-${LLM1_MODEL_TYPE_DEFAULT}}
LLM1_TEMPERATURE=${TEMPERATURE:-${LLM1_TEMPERATURE_DEFAULT}}
LLM1_MAX_TOKENS=${MAX_TOKENS:-${LLM1_MAX_TOKENS_DEFAULT}}
LLM1_IS_MISTRAL=${LLM1_IS_MISTRAL_DEFAULT}

# LLM2 Configuration (Always from model_config.yml)
LLM2_MODEL_NAME=${LLM2_MODEL_NAME_DEFAULT}
LLM2_MODEL_TYPE=${LLM2_MODEL_TYPE_DEFAULT}
LLM2_TEMPERATURE=${LLM2_TEMPERATURE_DEFAULT}
LLM2_MAX_TOKENS=${LLM2_MAX_TOKENS_DEFAULT}
LLM2_IS_MISTRAL=${LLM2_IS_MISTRAL_DEFAULT}
EOF

echo ".env file has been generated successfully!"
