#!/bin/bash
set -e

# Use proper connection parameters inside container
POSTGRES_HOST=${POSTGRES_HOST:-localhost}

# Function to check if postgres is ready
wait_for_postgres() {
    until pg_isready -h "$POSTGRES_HOST" -U "$POSTGRES_USER"; do
        echo "Waiting for database to be ready..."
        sleep 2
    done
}

# Function to run initialization
initialize_db() {
    echo "Creating extensions and tables..."
    psql -v ON_ERROR_STOP=1 -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" <<-EOSQL
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

# Wait for PostgreSQL to be ready
wait_for_postgres

# Check if initialization is needed
echo "Checking if initialization is needed..."
if ! psql -h "$POSTGRES_HOST" -U "$POSTGRES_USER" -d "$POSTGRES_DB" -c "SELECT 1 FROM embeddings LIMIT 1" >/dev/null 2>&1; then
    echo "Initializing database..."
    initialize_db
else
    echo "Database already initialized"
fi