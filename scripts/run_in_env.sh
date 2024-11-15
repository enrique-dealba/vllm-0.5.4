#!/bin/bash
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

# Use conda python explicitly
PYTHON_PATH=$(conda run -n vllm which python)
export PYTHON_PATH

# Execute command
exec "$@"