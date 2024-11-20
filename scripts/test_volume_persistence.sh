#!/bin/bash
set -e

echo "Testing Database Volume Persistence"
echo "================================="

DB_USER=${POSTGRES_USER:-test_user}
DB_PASSWORD=${POSTGRES_PASSWORD:-test_password}
DB_NAME=${POSTGRES_DB:-test_db}
DB_PORT=5433

# Function to execute SQL with proper error handling
execute_sql() {
    local sql=$1
    PGPASSWORD=$DB_PASSWORD psql -v ON_ERROR_STOP=1 -h localhost -p $DB_PORT -U $DB_USER -d $DB_NAME -c "$sql"
}

# Function to wait for database
wait_for_db() {
    echo "Waiting for database to be ready..."
    for i in {1..30}; do
        if PGPASSWORD=$DB_PASSWORD psql -h localhost -p $DB_PORT -U $DB_USER -d $DB_NAME -c '\q' 2>/dev/null; then
            echo "Database is ready!"
            return 0
        fi
        echo "Waiting... ($i/30)"
        sleep 1
    done
    echo "Database failed to become ready"
    return 1
}

echo "1. Ensuring clean state..."
docker compose -f docker-compose.test.yml down --remove-orphans >/dev/null 2>&1
docker volume rm test_timescaledb_data >/dev/null 2>&1 || true

echo "2. Starting fresh database..."
docker compose -f docker-compose.test.yml up -d test_db
wait_for_db

echo "3. Creating tables..."
execute_sql "
    CREATE EXTENSION IF NOT EXISTS vector;
    CREATE EXTENSION IF NOT EXISTS timescaledb;
    
    CREATE TABLE IF NOT EXISTS embeddings (
        id UUID PRIMARY KEY,
        metadata JSONB,
        content TEXT,
        embedding vector(384),
        created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
    );
"

echo "4. Inserting test data..."
TIMESTAMP=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
execute_sql "
    INSERT INTO embeddings (id, metadata, content, embedding, created_at)
    VALUES (
        '123e4567-e89b-12d3-a456-426614174000'::uuid,
        '{\"source\": \"persistence_test\", \"timestamp\": \"$TIMESTAMP\", \"test_type\": \"shell_test\"}'::jsonb,
        'Persistence test content',
        array_fill(0.1, ARRAY[384])::vector,
        CURRENT_TIMESTAMP
    );
"

echo "5. Verifying initial insertion..."
INITIAL_COUNT=$(execute_sql "SELECT COUNT(*) FROM embeddings;" | grep -oE '[0-9]+')
echo "Initial record count: $INITIAL_COUNT"

if [ "$INITIAL_COUNT" -ne "1" ]; then
    echo "ERROR: Expected 1 record, found $INITIAL_COUNT"
    exit 1
fi

echo "6. Stopping containers..."
docker compose -f docker-compose.test.yml stop test_db

echo "7. Starting containers..."
docker compose -f docker-compose.test.yml start test_db
wait_for_db

echo "8. Verifying data persistence..."
FINAL_COUNT=$(execute_sql "SELECT COUNT(*) FROM embeddings;" | grep -oE '[0-9]+')
echo "Final record count: $FINAL_COUNT"

if [ "$FINAL_COUNT" -ne "$INITIAL_COUNT" ]; then
    echo "ERROR: Data persistence failed. Expected $INITIAL_COUNT record(s), found $FINAL_COUNT"
    exit 1
fi

# Additional verification of data integrity
echo "9. Verifying data integrity..."
DATA_VERIFICATION=$(execute_sql "
    SELECT 
        id::text,
        metadata->>'source' as source,
        content
    FROM embeddings
    WHERE id = '123e4567-e89b-12d3-a456-426614174000'::uuid;
")

echo "Data verification result:"
echo "$DATA_VERIFICATION"

echo "SUCCESS: Data persistence verified!"