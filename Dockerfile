# 使用 Debian-slim 镜像以支持 agy 的 glibc 依赖
FROM python:3.11-slim

# 安装运行时必需的最小依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    curl \
    git \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# 确保 HOME 环境变量在所有场景下可用（包括 docker exec 进入的非登录 shell）
ENV HOME=/root

# 安装 agy CLI 并创建全局软链接
RUN curl -fsSL https://antigravity.google/cli/install.sh | bash \
    && ln -s /root/.local/bin/agy /usr/local/bin/agy

# 获取 uv 工具
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# 工作目录
WORKDIR /app

# 复制依赖清单与 README.md (hatchling 构建元数据需要)
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-install-project --no-dev

# 复制源码并安装项目本身，使镜像自包含
COPY src /app/src
RUN uv sync --frozen --no-dev

# 启动脚本
COPY entrypoint.sh /entrypoint.sh

EXPOSE 8765
ENTRYPOINT ["/bin/bash", "/entrypoint.sh"]
