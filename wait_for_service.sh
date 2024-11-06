#!/bin/bash

wait_for_service() {
    local service=$1
    local port=$2
    local max_attempts=10
    local wait_seconds=4

    for ((i=1; i<=max_attempts; i++)); do
        if curl -s "http://localhost:${port}/health" | grep -q '"status":"healthy"'; then
            echo "$service is healthy"
            return 0
        fi
        echo "Waiting for $service to become healthy (attempt $i/$max_attempts)..."
        sleep $wait_seconds
    done
    echo "Error: $service failed to become healthy after $max_attempts attempts"
    return 1
}

# Wait for main_server
wait_for_service "main_server" 8888

# Wait for llm1
wait_for_service "llm1" 8881

# Wait for llm2
wait_for_service "llm2" 8882

echo "All services are healthy!"
