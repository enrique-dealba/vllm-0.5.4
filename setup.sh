#!/bin/bash

# Default values
DEFAULT_DB_USER="postgres"
DEFAULT_DB_PASSWORD="password"

usage() {
    echo "Usage: $0 [options]"
    echo "Options:"
    echo "  -u, --user       Database user (default: postgres)"
    echo "  -p, --password   Database password (default: password)"
    echo "  -h, --help       Display this help message"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -u|--user)
            DB_USER="$2"
            shift 2
            ;;
        -p|--password)
            DB_PASSWORD="$2"
            shift 2
            ;;
        -h|--help)
            usage
            ;;
        *)
            echo "Unknown option: $1"
            usage
            ;;
    esac
done

DB_USER=${DB_USER:-$DEFAULT_DB_USER}
DB_PASSWORD=${DB_PASSWORD:-$DEFAULT_DB_PASSWORD}

# Clean up
echo "Cleaning up existing Docker resources..."
docker compose down -v 2>/dev/null || true
docker rm -f timescaledb 2>/dev/null || true
docker volume rm timescaledb_data 2>/dev/null || true

# Generate .env file
echo "Generating .env file..."
cat > .env << EOF
POSTGRES_DB=postgres
POSTGRES_USER=$DB_USER
POSTGRES_PASSWORD=$DB_PASSWORD
EOF

echo "Environment file created successfully!"

# Docker Compose
echo "Starting Docker Compose..."
docker compose up -d

# Check if containers are running
if [ $? -eq 0 ]; then
    echo "TimescaleDB is now running!"
    echo "You can connect to the database using:"
    echo "  Host: localhost"
    echo "  Port: 5432"
    echo "  User: $DB_USER"
    echo "  Password: $DB_PASSWORD"
else
    echo "Error: Failed to start Docker Compose"
    exit 1
fi
