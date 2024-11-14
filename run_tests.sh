#!/bin/bash

# Stop any existing containers
docker-compose -f docker-compose.test.yml down -v

# Build and run tests
docker-compose -f docker-compose.test.yml build
docker-compose -f docker-compose.test.yml run tests

# Cleanup
docker-compose -f docker-compose.test.yml down -v