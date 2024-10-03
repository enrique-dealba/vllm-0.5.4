#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/bin/activate vllm

# Determine run mode
RUN_MODE=${RUN_MODE:-server}

# Set PYTHONPATH
export PYTHONPATH="/app:$PYTHONPATH"

if [ "$RUN_MODE" = "server" ]; then
    echo "Starting FastAPI server..."
    exec uvicorn app.langchain_server:app --host 0.0.0.0 --port ${PORT:-8888} --workers 1
elif [ "$RUN_MODE" = "ui" ]; then
    echo "Starting Streamlit UI..."
    exec streamlit run /app/app/streamlit_stream.py --server.port ${PORT:-8888} --server.address 0.0.0.0
else
    echo "Invalid RUN_MODE: $RUN_MODE. Must be 'server' or 'ui'."
    exit 1
fi
