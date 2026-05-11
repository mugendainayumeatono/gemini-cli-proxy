#!/bin/bash

# test_api.sh
# 用于测试 Gemini CLI Proxy 提供的 OpenAI 兼容 API 是否可用

BASE_URL="http://127.0.0.1:8765"

# 尝试寻找一个 JSON 格式化工具让输出更好看
if command -v jq >/dev/null 2>&1; then
    JSON_FORMATTER="jq ."
elif command -v python3 >/dev/null 2>&1; then
    JSON_FORMATTER="python3 -m json.tool"
else
    JSON_FORMATTER="cat"
fi

echo "========================================="
echo "1. 检查服务健康状态 (GET /health)"
echo "========================================="
curl -s -X GET "$BASE_URL/health" | eval $JSON_FORMATTER
echo -e "\n"

echo "========================================="
echo "2. 获取可用模型列表 (GET /v1/models)"
echo "========================================="
curl -s -X GET "$BASE_URL/v1/models" -H "Authorization: Bearer sk-proxy-920cd156c9ec2f2616385cf7b7b5efe7" | eval $JSON_FORMATTER
echo -e "\n"

echo "========================================="
echo "3. 测试对话功能 [非流式] (POST /v1/chat/completions)"
echo "========================================="
# 注意：代码里没有强制要求 Authorization 请求头，但兼容 OpenAI 规范的客户端通常会发送
curl -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proxy-920cd156c9ec2f2616385cf7b7b5efe7" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {
        "role": "user",
        "content": "你好，请用一句话简短地介绍一下自己。"
      }
    ]
  }' | eval $JSON_FORMATTER
echo -e "\n"

echo "========================================="
echo "4. 测试对话功能 [流式打字机效果] (POST /v1/chat/completions)"
echo "========================================="
# 流式请求（Server-Sent Events），使用 -N 参数保持连接并实时打印
curl -N -s -X POST "$BASE_URL/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-proxy-920cd156c9ec2f2616385cf7b7b5efe7" \
  -d '{
    "model": "gemini-2.5-flash",
    "messages": [
      {
        "role": "user",
        "content": "请从 1 数到 5，不要有额外的解释。"
      }
    ],
    "stream": true
  }'
echo -e "\n\n========================================="
echo "✅ API 测试脚本执行完毕！"
