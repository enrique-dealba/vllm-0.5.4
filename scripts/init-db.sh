#!/bin/bash
set -e

echo "Starting database initialization..."

# Function to check if database is already initialized
check_initialization() {
    if psql -d "$POSTGRES_DB" -U "$POSTGRES_USER" -c "SELECT 1 FROM embeddings LIMIT 1" &>/dev/null; then
        echo "Database already initialized, skipping initialization"
        return 0
    fi
    return 1
}

# Only proceed with initialization if not already initialized
if ! check_initialization; then
    echo "Performing fresh initialization..."
    
    # Your existing initialization code here
    execute_sql "CREATE EXTENSION IF NOT EXISTS vector;" "Create vector extension"
    execute_sql "CREATE EXTENSION IF NOT EXISTS timescaledb;" "Create TimescaleDB extension"
    
    execute_sql "
        CREATE TABLE IF NOT EXISTS embeddings (
            id UUID PRIMARY KEY,
            metadata JSONB NOT NULL,
            content TEXT NOT NULL,
            embedding VECTOR(384) NOT NULL,
            created_at TIMESTAMPTZ DEFAULT CURRENT_TIMESTAMP
        );
    " "Create embeddings table"
    
    # Create indexes only if they don't exist
    execute_sql "
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 'idx_embeddings_metadata'
            ) THEN
                CREATE INDEX idx_embeddings_metadata ON embeddings USING GIN (metadata);
            END IF;
        END $$;
    " "Create metadata index"

    execute_sql "
        DO $$
        BEGIN
            IF NOT EXISTS (
                SELECT 1 FROM pg_indexes WHERE indexname = 'idx_embeddings_embedding'
            ) THEN
                CREATE INDEX idx_embeddings_embedding 
                ON embeddings USING ivfflat (embedding vector_cosine_ops)
                WITH (lists = 100);
            END IF;
        END $$;
    " "Create embedding index"
else
    echo "Skipping initialization as database is already set up"
fi