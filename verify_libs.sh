#!/bin/bash
set -x

echo "Library Paths:"
echo $LD_LIBRARY_PATH
echo

echo "Checking libffi:"
ldconfig -p | grep libffi
ldd /usr/lib/x86_64-linux-gnu/libffi.so.7

echo "Checking p11-kit:"
ldconfig -p | grep p11-kit
ldd /usr/lib/x86_64-linux-gnu/libp11-kit.so.0

echo "Testing psycopg2..."
python3 -c "import psycopg2; print('psycopg2 version:', psycopg2.__version__)"