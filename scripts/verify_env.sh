#!/bin/bash

echo "Environment Verification"
echo "======================"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo
echo "Library Locations:"
ldconfig -p | grep libffi
echo
echo "Testing psycopg2 import..."
python3 -c "import psycopg2; print('psycopg2 imported successfully')"