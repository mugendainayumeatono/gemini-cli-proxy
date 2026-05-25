#!/bin/bash

# 获取当前日期，格式为 YYYYMMDD
TAG_DATE=$(date +%Y%m%d)
IMAGE_NAME="gemini-cli-proxy"
IMAGE_TAG="agy_${TAG_DATE}"
FULL_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"
LATEST_IMAGE_NAME="${IMAGE_NAME}:latest"

echo "===================================================="
echo "🚀 开始构建 Docker 镜像: $FULL_IMAGE_NAME"
echo "===================================================="

# 执行 docker build
docker build -t "$FULL_IMAGE_NAME" .

if [ $? -eq 0 ]; then
    echo "===================================================="
    echo "🏷️  正在为最新镜像打上 latest 标签..."
    docker tag "$FULL_IMAGE_NAME" "$LATEST_IMAGE_NAME"
    
    echo "===================================================="
    echo "✅ 镜像构建成功！"
    echo "已生成标签: $IMAGE_TAG"
    echo "已生成标签: latest"
    echo "你可以直接使用 'docker compose up' 启动 latest 版本。"
    echo "===================================================="
else
    echo "===================================================="
    echo "❌ 镜像构建失败！"
    echo "===================================================="
    exit 1
fi
