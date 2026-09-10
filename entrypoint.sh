#!/bin/bash
set -e

WORKSPACE_DIR="/app/workspace"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

CONFIG_DIR="/root/.gemini"
LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/proxy.log"

mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"

check_agy() {
    echo "正在验证 Antigravity 连通性..." | tee -a "$LOG_FILE"
    
    # -p 模式下测试连通性，记录返回状态和退出码
    local exit_code=0
    agy -p "Reply with 'pong' and nothing else. DO NOT use tools." >>"$LOG_FILE" 2>&1 || exit_code=$?
    if [ $exit_code -eq 0 ]; then
        echo "Antigravity 连通性验证成功 (exit code: 0)" | tee -a "$LOG_FILE"
        return 0
    else
        echo "Antigravity 连通性验证失败 (exit code: $exit_code)" | tee -a "$LOG_FILE"
        return 1
    fi
}

if [ ! -f "$CONFIG_DIR/antigravity-cli/antigravity-oauth-token" ]; then
    {
        echo "===================================================="
        echo "❌ 尚未初始化（未找到授权文件）。"
        echo "为了避免 Docker Compose 日志前缀干扰界面，已取消自动运行配置。"
        echo "请按以下步骤完成初始化："
        echo "1. 保持当前容器运行（或使用 docker compose up -d 在后台运行）"
        echo "2. 在宿主机新开一个终端，执行 ./enter.sh 进入容器"
        echo "3. 在容器的原生 bash shell 中执行 'agy' 完成配置"
        echo "4. 配置完成后，重启本容器即可。"
        echo "===================================================="
    } | tee -a "$LOG_FILE"
    
    # 保持容器存活，以便用户可以通过 enter.sh 进入容器配置
    exec tail -f /dev/null
else
    echo "检测到已验证的配置，正在启动 Gemini CLI Proxy..." | tee -a "$LOG_FILE"
    if ! check_agy; then
        echo "⚠️ 警告：当前配置验证失败，请考虑执行 'docker compose down -v' 重新配置。" | tee -a "$LOG_FILE"
    fi
    
    echo "日志路径: $LOG_FILE" | tee -a "$LOG_FILE"
    exec uv run gemini-cli-proxy --host 0.0.0.0 --port 8765 2>&1 | tee -a "$LOG_FILE"
fi
