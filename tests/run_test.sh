#!/bin/bash
export PROXY_API_KEY="sk-proxy-920cd156c9ec2f2616385cf7b7b5efe7"
uv run gemini-cli-proxy &
SERVER_PID=$!
sleep 3
bash test_api.sh
kill $SERVER_PID
