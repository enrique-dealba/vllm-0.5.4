#!/bin/bash

# Initialize variables
HF_TOKEN=""
LANGCHAIN_KEY=""

# Function to display usage
usage() {
   echo "Usage: $0 -h <Hugging Face token> -l <LangChain API key>"
   echo "  -h    Hugging Face token"
   echo "  -l    LangChain API key"
   exit 1
}

# Parse command line arguments
while getopts "h:l:" opt; do
   case $opt in
       h) HF_TOKEN="$OPTARG" ;;
       l) LANGCHAIN_KEY="$OPTARG" ;;
       ?) usage ;;
   esac
done

# Check if both arguments are provided
if [ -z "$HF_TOKEN" ] || [ -z "$LANGCHAIN_KEY" ]; then
   usage
fi

# Create or overwrite .env file
echo "Generating .env file..."
cat > .env << EOF
HUGGING_FACE_HUB_TOKEN=$HF_TOKEN
LANGCHAIN_API_KEY=$LANGCHAIN_KEY
EOF

echo ".env file created successfully!"
echo "Contents:"
cat .env
