#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/bin/activate vllm

# Determine run mode
RUN_MODE=${RUN_MODE:-server}

# Set PYTHONPATH
export PYTHONPATH="/app:$PYTHONPATH"

case "$RUN_MODE" in
    server)
        echo "Starting FastAPI server..."
        exec uvicorn app.langchain_server:app --host 0.0.0.0 --port ${PORT:-8888} --workers 1
        ;;
    ui)
        echo "Starting marimo UI..."
        exec marimo run /app/app/marimo_ui.py --host 0.0.0.0 --port ${PORT:-8888}
        ;;
    tests)
        echo "Running tests..."
        PYTHONPATH=/app pytest /app/tests/ -v --cov=app "$@"
        ;;
    *)
        echo "Invalid RUN_MODE: $RUN_MODE. Must be 'server', 'ui', or 'tests'."
        exit 1
        ;;
esac