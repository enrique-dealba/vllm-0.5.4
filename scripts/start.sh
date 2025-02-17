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
    analysis)
        echo "Running LLM interpretability analysis..."
        exec python -m app.interpretability_analysis
        ;;
    ui)
        echo "Starting Streamlit UI..."
        exec streamlit run /app/app/streamlit_ui.py --server.port ${PORT:-8888} --server.address 0.0.0.0
        ;;
    objectives)
        echo "Starting Streamlit UI..."
        exec streamlit run /app/app/objectives_ui.py --server.port ${PORT:-8888} --server.address 0.0.0.0
        ;;
    intents)
        echo "Starting Streamlit UI for Intents..."
        exec streamlit run /app/app/streamlit_intents.py --server.port ${PORT:-8888} --server.address 0.0.0.0
        ;;
    rag)
        echo "Starting Streamlit UI for RAG..."
        exec streamlit run /app/app/chunk_ui.py --server.port ${PORT:-8888} --server.address 0.0.0.0
        ;;
    tests)
        echo "Running tests..."
        PYTHONPATH=/app pytest /app/tests/ -v -s --cov=app "$@"
        ;;
    *)
        echo "Invalid RUN_MODE: $RUN_MODE. Must be 'server', 'ui', 'intents', 'rag', or 'tests'."
        exit 1
        ;;
esac