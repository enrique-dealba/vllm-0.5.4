#!/bin/bash
set -e

echo "Testing Database Volume Persistence"
echo "================================="

DB_USER=${POSTGRES_USER:-test_user}
DB_PASSWORD=${POSTGRES_PASSWORD:-test_password}
DB_NAME=${POSTGRES_DB:-test_db}
DB_PORT=5433  # Test database port

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

# Function to check record count
check_count() {
    PGPASSWORD=$DB_PASSWORD psql -h localhost -p $DB_PORT -U $DB_USER -d $DB_NAME -t -c "SELECT COUNT(*) FROM embeddings;"
}

echo "1. Ensuring clean state..."
docker compose -f docker-compose.test.yml down -v >/dev/null 2>&1
docker volume rm test_timescaledb_data >/dev/null 2>&1 || true

echo "2. Starting fresh database..."
docker compose -f docker-compose.test.yml up -d test_db
wait_for_db

echo "3. Inserting test data..."
PGPASSWORD=$DB_PASSWORD psql -h localhost -p $DB_PORT -U $DB_USER -d $DB_NAME << EOF
INSERT INTO embeddings (id, metadata, content, embedding, created_at)
VALUES (
    '123e4567-e89b-12d3-a456-426614174000',
    '{"source": "persistence_test", "timestamp": "'$(date -u +"%Y-%m-%dT%H:%M:%SZ")'"}',
    'Persistence test content',
    array_fill(0.1, ARRAY[384])::vector,
    CURRENT_TIMESTAMP
);
EOF

echo "4. Verifying initial insertion..."
initial_count=$(check_count)
echo "Initial record count: $initial_count"
if [ "$initial_count" -ne 1 ]; then
    echo "ERROR: Expected 1 record, found $initial_count"
    exit 1
fi

echo "5. Stopping containers..."
docker compose -f docker-compose.test.yml stop test_db

echo "6. Starting containers..."
docker compose -f docker-compose.test.yml start test_db
wait_for_db

echo "7. Verifying data persistence..."
final_count=$(check_count)
echo "Final record count: $final_count"
if [ "$final_count" -ne "$initial_count" ]; then
    echo "ERROR: Data persistence failed. Expected $initial_count record(s), found $final_count"
    exit 1
fi

echo "SUCCESS: Data persistence verified!"