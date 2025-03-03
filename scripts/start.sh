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
    objectives)
        echo "Starting Streamlit UI..."
        exec streamlit run /app/app/objectives_ui.py --server.port ${PORT:-8888} --server.address 0.0.0.0
        ;;
    tests)
        echo "Running tests..."
        PYTHONPATH=/app pytest /app/tests/ -v -s --cov=app "$@"
        ;;
    run_experiment)
        echo "Running full experiment workflow..."
        echo "Starting FastAPI server..."
        uvicorn app.langchain_server:app --host 0.0.0.0 --port ${PORT:-8888} --workers 1 &
        echo "Waiting 45 seconds for server warm-up..."
        sleep 45
        echo "Starting experiment workflow..."
        if [ -n "${TEST_CASE}" ]; then
            echo "Running with TEST_CASE=${TEST_CASE}"
            python -m app.run_experiment --input_text "${INPUT_TEXT}" --iterations "${ITERATIONS:-1}" --test_case "${TEST_CASE}"
        else
            python -m app.run_experiment --input_text "${INPUT_TEXT}" --iterations "${ITERATIONS:-1}"
        fi
        ;;
    *)
        echo "Invalid RUN_MODE: $RUN_MODE. Must be one of 'server', 'analysis', 'objectives', 'tests', or 'run_experiment'."
        exit 1
        ;;
esac