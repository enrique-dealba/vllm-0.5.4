#!/bin/bash
set -e

# Database connection details from environment variables
DB_HOST=${DB_HOST:-localhost}
DB_PORT=${DB_PORT:-5432}
DB_NAME=${POSTGRES_DB:-postgres}
DB_USER=${POSTGRES_USER:-postgres}
DB_PASSWORD=${POSTGRES_PASSWORD:-password}

# Export PGPASSWORD to avoid password prompt
export PGPASSWORD="$DB_PASSWORD"

# Function to execute a SQL command and capture output
execute_sql() {
    local sql="$1"
    psql -h "$DB_HOST" -p "$DB_PORT" -U "$DB_USER" -d "$DB_NAME" -c "$sql"
}

# Check for embeddings table
echo "Checking for 'embeddings' table..."
EMBEDDINGS_PRESENT=$(execute_sql "\dt embeddings" | grep embeddings || true)
if [ -n "$EMBEDDINGS_PRESENT" ]; then
    echo "SUCCESS: 'embeddings' table exists."
    exit 0
else
    echo "FAILURE: 'embeddings' table does NOT exist."
    exit 1
fi
