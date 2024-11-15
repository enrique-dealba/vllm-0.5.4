#!/bin/bash

echo "Checking library versions and symbols..."

# Check libffi
echo "libffi details:"
ldconfig -p | grep libffi
nm -D /usr/lib/x86_64-linux-gnu/libffi.so.7 | grep ffi_type_pointer

# Check psycopg2 installation
echo -e "\npsycopg2 installation:"
python3 -c "import psycopg2; print('psycopg2 version:', psycopg2.__version__)"

# List all loaded libraries
echo -e "\nLoaded libraries:"
ldd $(python3 -c "import psycopg2._psycopg; print(psycopg2._psycopg.__file__)")