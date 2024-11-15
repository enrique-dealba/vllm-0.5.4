#!/bin/bash

# Enable error tracing
set -x
set -e

echo "==========================="
echo "Starting Library Verification"
echo "==========================="

echo "Current working directory:"
pwd

echo "==========================="
echo "Library Paths:"
echo $LD_LIBRARY_PATH
echo "==========================="

echo "Checking libffi:"
if ! ldconfig -p | grep libffi; then
    echo "ERROR: libffi not found!"
    exit 1
fi
if ! ldd /usr/lib/x86_64-linux-gnu/libffi.so.7; then
    echo "ERROR: Cannot check libffi.so.7 dependencies!"
    exit 1
fi

echo "==========================="
echo "Checking p11-kit:"
if ! ldconfig -p | grep p11-kit; then
    echo "ERROR: p11-kit not found!"
    exit 1
fi
if ! ldd /usr/lib/x86_64-linux-gnu/libp11-kit.so.0; then
    echo "ERROR: Cannot check libp11-kit.so.0 dependencies!"
    exit 1
fi

echo "==========================="
echo "Testing psycopg2..."
if ! python3 -c "import psycopg2; print('psycopg2 version:', psycopg2.__version__)"; then
    echo "ERROR: psycopg2 import failed!"
    exit 1
fi

echo "==========================="
echo "All verifications passed!"
echo "==========================="