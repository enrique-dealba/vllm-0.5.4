#!/bin/bash
set -e

# Check if database is already initialized by looking for our custom marker
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "SELECT 1 FROM pg_tables WHERE tablename = 'embeddings'" > /dev/null 2>&1
DB_INITIALIZED=$?

if [ $DB_INITIALIZED -eq 0 ]; then
    echo "Database already initialized, skipping initialization"
    exit 0
fi

echo "Initializing database..."
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Install required extensions
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS timescaledb;
EOSQL

echo "Database initialization complete"