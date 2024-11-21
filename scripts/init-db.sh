#!/bin/bash
set -e

# Function to run initialization
initialize_db() {
    echo "Initializing database..."
    psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
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

# Try to check if database exists and is accessible
until pg_isready -U "$POSTGRES_USER" -d "$POSTGRES_DB" -h localhost -p 5432; do
    echo "Waiting for database to be ready..."
    sleep 2
done

# Check if initialization is needed
if ! psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" -c "SELECT 1 FROM embeddings LIMIT 1" >/dev/null 2>&1; then
    initialize_db
else
    echo "Database already initialized and embeddings table exists"
fi