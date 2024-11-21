#!/bin/bash
set -e

# Wait for PostgreSQL to be ready
until pg_isready -q; do
    echo "Waiting for PostgreSQL to be ready..."
    sleep 1
done

# Function to run initialization
initialize_db() {
    echo "Running database initialization..."
    # Use socket connection by default
    psql -v ON_ERROR_STOP=1 <<-EOSQL
        -- Install required extensions
        CREATE EXTENSION IF NOT EXISTS vector;
        CREATE EXTENSION IF NOT EXISTS timescaledb;

        -- Create embeddings table
        CREATE TABLE IF NOT EXISTS embeddings (
            id UUID PRIMARY KEY,
            metadata JSONB NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );

        -- Create indexes
        CREATE INDEX IF NOT EXISTS idx_embeddings_metadata 
        ON embeddings USING GIN (metadata);

        CREATE INDEX IF NOT EXISTS idx_embeddings_embedding 
        ON embeddings USING ivfflat (embedding vector_cosine_ops)
        WITH (lists = 100);
EOSQL
    echo "Database initialization complete"
}

# Check if initialization is needed
if ! psql -q -c "SELECT 1 FROM embeddings LIMIT 1" >/dev/null 2>&1; then
    initialize_db
else
    echo "Database already initialized"
fi