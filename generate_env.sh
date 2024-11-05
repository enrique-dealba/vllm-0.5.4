#!/bin/bash

# Check if correct number of arguments is provided
if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <huggingface_token> <langchain_api_key>"
    exit 1
fi

# Create .env file with the provided arguments
cat > .env << EOF
# .env
HUGGING_FACE_HUB_TOKEN=$1
LANGCHAIN_API_KEY=$2
EOF

echo ".env file has been generated successfully!"
