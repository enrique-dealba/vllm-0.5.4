#!/bin/bash
set -e

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

# Get conda python path
export CONDA_PYTHON=$(which python)
echo "Using Python: $CONDA_PYTHON"

# Execute command with conda python
exec "$@"