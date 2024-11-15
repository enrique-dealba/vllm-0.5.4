#!/bin/bash
set -x

echo "=== Library Verification Start ==="

# Activate conda environment
source /root/miniconda3/bin/activate vllm

echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
ldconfig -p | grep libffi
ldconfig -p | grep p11-kit
ldd /usr/lib/x86_64-linux-gnu/libffi.so.7
ldd /usr/lib/x86_64-linux-gnu/libp11-kit.so.0

echo "=== Testing psycopg2 ==="
python3 -c "import psycopg2; print('psycopg2 version:', psycopg2.__version__)"

if [ $? -ne 0 ]; then
    echo "ERROR: psycopg2 verification failed"
    exit 1
fi

echo "=== Testing pluggy ==="
python3 -c 'import pluggy; print("pluggy version:", pluggy.__version__)'

if [ $? -ne 0 ]; then
    echo "ERROR: pluggy verification failed"
    exit 1
fi

echo "All verifications passed!"