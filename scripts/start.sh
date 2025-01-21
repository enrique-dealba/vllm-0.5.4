#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/bin/activate vllm

# Print package versions
echo "Installed package versions:"
while IFS= read -r package; do
    if [ ! -z "$package" ]; then
        pip freeze | grep -i "^${package}="
    fi
done < requirements.txt

# Determine run mode
RUN_MODE=${RUN_MODE:-server}

# Set PYTHONPATH
export PYTHONPATH="/app:$PYTHONPATH"

if [ "$RUN_MODE" = "server" ]; then
    echo "Starting FastAPI server..."
    exec uvicorn app.langchain_server:app --host 0.0.0.0 --port ${PORT:-8888} --workers 1
elif [ "$RUN_MODE" = "ui" ]; then
    echo "Starting Streamlit UI..."
    exec streamlit run /app/app/streamlit_ui.py --server.port ${PORT:-8888} --server.address 0.0.0.0
elif [ "$RUN_MODE" = "intents" ]; then
    echo "Starting Streamlit UI for Intents..."
    exec streamlit run /app/app/streamlit_intents.py --server.port ${PORT:-8888} --server.address 0.0.0.0
elif [ "$RUN_MODE" = "rag" ]; then
    echo "Starting Streamlit UI for RAG..."
    exec streamlit run /app/app/chunk_ui.py --server.port ${PORT:-8888} --server.address 0.0.0.0
else
    echo "Invalid RUN_MODE: $RUN_MODE. Must be 'server', 'ui', 'intents', or 'rag'."
    exit 1
fi
