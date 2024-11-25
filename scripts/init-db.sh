#!/bin/bash
set -e

# Make sure this aligns with the PGDATA mapping
INIT_MARKER="/var/lib/postgresql/data/.initialized"

# Check if already initialized
if [ -f "$INIT_MARKER" ]; then
    echo "Database already initialized, skipping..."
    exit 0
fi

echo "Starting database initialization..."

# Function to execute SQL with error handling
execute_sql() {
    local sql="$1"
    local msg="$2"
    echo "Executing: $msg"
    if ! psql -v ON_ERROR_STOP=1 -d "$POSTGRES_DB" -U "$POSTGRES_USER" -c "$sql"; then
        echo "Error executing SQL: $msg"
        return 1
    fi
    return 0
}

# Function to check if database is already initialized
check_initialization() {
    if psql -d "$POSTGRES_DB" -U "$POSTGRES_USER" -tc "SELECT 1 FROM pg_tables WHERE tablename = 'embeddings'" | grep -q 1; then
        echo "Database already initialized"
        return 0
    fi
    return 1
}

# Main initialization logic
main() {
    # Create extensions first
    execute_sql "CREATE EXTENSION IF NOT EXISTS vector;" "Create vector extension"
    execute_sql "CREATE EXTENSION IF NOT EXISTS timescaledb;" "Create TimescaleDB extension"

    # Only create table if it doesn't exist
    if ! check_initialization; then
        echo "Creating embeddings table..."
        execute_sql "
            CREATE TABLE IF NOT EXISTS embeddings (
                id UUID PRIMARY KEY,
                metadata JSONB NOT NULL,
                content TEXT NOT NULL,
                embedding VECTOR(384) NOT NULL,
                created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
            );
        " "Create embeddings table"

        echo "Creating indexes..."
        execute_sql "
            CREATE INDEX IF NOT EXISTS idx_embeddings_metadata 
            ON embeddings USING GIN (metadata);
        " "Create metadata index"

        execute_sql "
            CREATE INDEX IF NOT EXISTS idx_embeddings_embedding 
            ON embeddings USING ivfflat (embedding vector_cosine_ops)
            WITH (lists = 100);
        " "Create embedding index"

        echo "Initialization complete"
    else
        echo "Table already exists, skipping initialization"
    fi
}

# Execute main logic
main

touch "$INIT_MARKER"