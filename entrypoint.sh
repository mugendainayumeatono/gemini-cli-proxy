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
    echo "正在验证 Antigravity 连通性 (使用 policies/no-tools.json 限制工具)..."
    
    # 使用项目内预设的 Policy 文件限制所有工具调用
    # -p 模式下无法交互审批，具有限制工具的效果
    if agy --policy "/app/policies/no-tools.json" -p "Reply with 'pong' and nothing else. DO NOT use tools." >"$LOG_FILE" 2>&1; then
        return 0
    else
        return 1
    fi
}

# 判断是否已初始化的条件：存在 oauth_creds.json 或 gemini-credentials.json
if [ ! -f "$CONFIG_DIR/oauth_creds.json" ] && [ ! -f "$CONFIG_DIR/gemini-credentials.json" ]; then
    echo "===================================================="
    echo "检测到尚未初始化（未找到授权文件），即将进入交互式配置。"
    echo "⚠️ 正在清理终端状态与残留缓冲，请等待 1 秒..."
    echo "===================================================="
    
    # 等待 Docker Compose 的 TTY 附加操作完全完成
    sleep 1
    
    # 尝试重置终端为正常模式，防止之前的命令或 Docker 导致 TTY 状态异常
    stty sane 2>/dev/null || true
    
    # 强力清空输入缓冲区中的残留字符（包括按键缓存、终端光标响应等）
    while read -r -t 0.1 -n 10000; do :; done
    
    # 1. 运行交互式初始化
    agy
    
    # 2. 立即进行功能性验证
    if check_agy; then
        echo "===================================================="
        echo "✅ 验证成功！Antigravity 已准备就绪。"
        echo "配置已保存。请按 Ctrl+C 停止当前容器，"
        echo "然后运行 'docker compose up -d' 以正常启动代理服务。"
        echo "===================================================="
        exit 0
    else
        echo "===================================================="
        echo "❌ 错误：验证失败！'agy -p' 无法获得正常响应。"
        echo "可能原因：API Key 无效、配额超限或网络连接问题。"
        echo "请检查网络或重新运行配置流程。"
        echo "===================================================="
        exit 1
    fi
else
    echo "检测到已验证的配置，正在启动 Gemini CLI Proxy..."
    if ! check_agy; then
        echo "⚠️ 警告：当前配置验证失败，请考虑执行 'docker compose down -v' 重新配置。"
    fi
    
    echo "日志路径: $LOG_FILE"
    exec uv run gemini-cli-proxy --host 0.0.0.0 --port 8765 2>&1 | tee -a "$LOG_FILE"
fi
