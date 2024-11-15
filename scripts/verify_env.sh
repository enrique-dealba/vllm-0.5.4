#!/bin/bash
set -e

echo "Environment Verification"
echo "======================"

# Activate conda environment
eval "$(conda shell.bash hook)"
conda activate vllm

# Get conda Python path
CONDA_PYTHON=$(which python)
echo "Using Python: $CONDA_PYTHON"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo

echo "Library Locations:"
ldconfig -p | grep libffi
echo

echo "Testing psycopg2 import..."
$CONDA_PYTHON -c "import psycopg2; print('psycopg2 imported successfully')"