#!/bin/bash
set -e

echo "Starting database initialization..."

# Function to wait for PostgreSQL
wait_for_postgres() {
    until pg_isready -q; do
        echo "Waiting for PostgreSQL..."
        sleep 1
    done
}

# Configure PostgreSQL for external connections
configure_postgres() {
    echo "Configuring PostgreSQL..."
    psql -v ON_ERROR_STOP=1 <<-EOSQL
        ALTER SYSTEM SET listen_addresses TO '*';
        ALTER SYSTEM SET max_connections TO '100';
EOSQL
    # Ensure pg_hba.conf allows connections
    echo "host all all all md5" >> /home/postgres/pgdata/data/pg_hba.conf
}

# Initialize database objects
initialize_db() {
    echo "Creating database objects..."
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
}

# Main initialization sequence
main() {
    wait_for_postgres
    
    # Check if initialization is needed
    if ! psql -q -c "SELECT 1 FROM pg_tables WHERE tablename = 'embeddings'" >/dev/null 2>&1; then
        echo "Running full initialization..."
        configure_postgres
        initialize_db
        echo "Reloading PostgreSQL configuration..."
        pg_ctl reload
        echo "Database initialization complete"
    else
        echo "Database already initialized"
    fi
}

# Execute main function
main