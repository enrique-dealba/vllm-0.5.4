#!/bin/bash

echo "Cleaning up existing containers..."
docker compose -f docker-compose.test.yml down -v --remove-orphans

# Build and run tests
docker compose -f docker-compose.test.yml build
docker compose -f docker-compose.test.yml run tests

# Cleanup
docker compose -f docker-compose.test.yml down -v