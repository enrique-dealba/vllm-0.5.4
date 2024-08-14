#!/bin/bash
set -e

# Activate conda environment
source /root/miniconda3/bin/activate vllm

# Start the FastAPI server using uvicorn
exec uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-8888} --workers 1
