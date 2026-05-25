#!/bin/bash

CONTAINER_NAME="gemini-cli-proxy"

# 检查容器是否正在运行
if [ "$(docker ps -q -f name=^/${CONTAINER_NAME}$)" ]; then
    echo "===================================================="
    echo "✅ 发现正在运行的容器 $CONTAINER_NAME，正在进入..."
    echo "===================================================="
    # 使用 docker exec 交互式进入容器内的 bash 终端
    docker exec -it $CONTAINER_NAME /bin/bash
else
    echo "===================================================="
    echo "❌ 容器 $CONTAINER_NAME 未启动或不存在！"
    echo "你可以通过执行 'docker compose up -d' 启动它。"
    echo "===================================================="
    exit 1
fi
