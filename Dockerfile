# 使用超小规模的 Alpine 镜像
FROM python:3.11-alpine

# 安装运行时必需的最小依赖
RUN apk add --no-cache nodejs npm bash curl git

# 全局安装 gemini-cli 并清理缓存
RUN npm install -g @google/gemini-cli && npm cache clean --force

# 获取 uv 工具
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 工作目录
WORKDIR /app

# 复制并安装依赖
COPY pyproject.toml uv.lock ./
RUN uv sync --no-install-project

# 启动脚本
COPY entrypoint.sh /entrypoint.sh

EXPOSE 8765
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
