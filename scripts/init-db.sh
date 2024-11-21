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