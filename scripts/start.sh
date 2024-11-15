#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/bin/activate vllm

# Set PYTHONPATH
export PYTHONPATH="/app:$PYTHONPATH"

# # Function to check if the database is ready
# wait_for_db() {
#     echo "Waiting for TimescaleDB to be ready..."
#     while ! nc -z timescaledb 5432; do
#         echo "Database is not ready yet. Retrying in 2 seconds..."
#         sleep 2
#     done
#     echo "Database is up and running!"
# }

# # Call the wait function
# wait_for_db

# Determine run mode
RUN_MODE=${RUN_MODE:-server}

if [ "$RUN_MODE" = "server" ]; then
    echo "Starting FastAPI server..."
    exec uvicorn app.langchain_server:app --host 0.0.0.0 --port ${PORT:-8888} --workers 1
elif [ "$RUN_MODE" = "ui" ]; then
    echo "Starting Streamlit UI..."
    exec streamlit run /app/app/chunk_ui.py --server.port ${PORT:-8888} --server.address 0.0.0.0
elif [ "$RUN_MODE" = "test" ]; then
    echo "Running tests..."
    exec pytest tests/ -v --log-cli-level=INFO
else
    echo "Invalid RUN_MODE: $RUN_MODE. Must be 'server', 'ui', or 'test'."
    exit 1
fi
