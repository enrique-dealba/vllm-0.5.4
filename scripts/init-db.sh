#!/bin/bash
set -e

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

# Function to verify table exists
verify_table() {
    local table="$1"
    psql -d "$POSTGRES_DB" -U "$POSTGRES_USER" -c "\dt $table" 2>/dev/null | grep -q "$table"
    return $?
}

# Main initialization function
initialize_database() {
    echo "Creating extensions..."
    execute_sql "CREATE EXTENSION IF NOT EXISTS vector;" "Create vector extension" || return 1
    execute_sql "CREATE EXTENSION IF NOT EXISTS timescaledb;" "Create TimescaleDB extension" || return 1

    echo "Creating embeddings table..."
    execute_sql "
        DROP TABLE IF EXISTS embeddings;
        CREATE TABLE embeddings (
            id UUID PRIMARY KEY,
            metadata JSONB NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    " "Create embeddings table" || return 1

    echo "Creating indexes..."
    execute_sql "
        CREATE INDEX IF NOT EXISTS idx_embeddings_metadata 
        ON embeddings USING GIN (metadata);
    " "Create metadata index" || return 1

    execute_sql "
        CREATE INDEX IF NOT EXISTS idx_embeddings_embedding 
        ON embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
    " "Create embedding index" || return 1

    return 0
}

# Verify initialization
verify_initialization() {
    echo "Verifying initialization..."
    
    if ! verify_table "embeddings"; then
        echo "ERROR: embeddings table not found after initialization"
        return 1
    fi

    if ! execute_sql "SELECT COUNT(*) FROM embeddings LIMIT 1" "Verify embeddings table access"; then
        echo "ERROR: Cannot access embeddings table"
        return 1
    fi

    echo "Initialization verified successfully"
    return 0
}

# Main execution
echo "Waiting for PostgreSQL to be ready..."
until pg_isready -d "$POSTGRES_DB" -U "$POSTGRES_USER" -q; do
    echo "Waiting for database connection..."
    sleep 1
done

echo "Database is ready, proceeding with initialization..."
if initialize_database && verify_initialization; then
    echo "Database initialization completed successfully"
else
    echo "Database initialization failed"
    exit 1
fi