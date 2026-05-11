#!/bin/bash
uv run gemini-cli-proxy --port 8766 &
SERVER_PID=$!
sleep 2

echo "Testing without API Key..."
curl -s http://localhost:8766/v1/models | jq .
echo ""

echo "Testing with API Key..."
curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer test_key" http://localhost:8766/v1/models
echo ""

kill $SERVER_PID
