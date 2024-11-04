#!/bin/bash

echo "Testing health endpoint..."
curl -v http://localhost:8888/health

echo -e "\n\nTesting llm1 generate endpoint..."
curl -v -X POST "http://localhost:8888/llm1/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "What is 7+8?"}'

echo -e "\n\nTesting llm2 generate endpoint..."
curl -v -X POST "http://localhost:8888/llm2/generate" \
     -H "Content-Type: application/json" \
     -d '{"text": "Explain the theory of relativity."}'
