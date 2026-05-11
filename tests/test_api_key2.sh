#!/bin/bash
uv run gemini-cli-proxy --port 8766 --api-key my_secret_key &
SERVER_PID=$!
sleep 2

echo "Testing with correct API Key..."
curl -s -o /dev/null -w "%{http_code}" -H "Authorization: Bearer my_secret_key" http://localhost:8766/v1/models
echo ""

echo "Testing with wrong API Key..."
curl -s http://localhost:8766/v1/models -H "Authorization: Bearer wrong_key" | jq .
echo ""

kill $SERVER_PID
