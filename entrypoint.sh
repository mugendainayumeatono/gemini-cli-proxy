#!/bin/bash
set -e

WORKSPACE_DIR="/app/workspace"
mkdir -p "$WORKSPACE_DIR"
cd "$WORKSPACE_DIR"

CONFIG_DIR="/root/.gemini"
SUCCESS_MARKER="$CONFIG_DIR/.init_done"
LOG_DIR="/app/logs"
LOG_FILE="$LOG_DIR/proxy.log"

mkdir -p "$LOG_DIR"
mkdir -p "$CONFIG_DIR"

check_gemini() {
    echo "正在验证 Gemini 连通性 (使用 policies/no-tools.json 限制工具)..."
    
    # 使用项目内预设的 Policy 文件限制所有工具调用 (方案1)
    # 结合明确的 Prompt 指令 (方案2)
    # -p 模式下无法交互审批，具有方案3的约束效果
    if gemini --policy "/app/policies/no-tools.json" -p "Reply with 'pong' and nothing else. DO NOT use tools." >"$LOG_FILE" 2>&1; then
        return 0
    else
        return 1
    fi
}

if [ ! -f "$SUCCESS_MARKER" ]; then
    echo "===================================================="
    echo "检测到尚未初始化，即将进入交互式配置。"
    echo "⚠️ 正在清理终端状态与残留缓冲，请等待 1 秒..."
    echo "===================================================="
    
    # 等待 Docker Compose 的 TTY 附加操作完全完成
    sleep 1
    
    # 尝试重置终端为正常模式，防止之前的命令或 Docker 导致 TTY 状态异常
    stty sane 2>/dev/null || true
    
    # 强力清空输入缓冲区中的残留字符（包括按键缓存、终端光标响应等）
    while read -r -t 0.1 -n 10000; do :; done
    
    # 1. 运行交互式初始化
    gemini
    
    # 2. 立即进行功能性验证
    if check_gemini; then
        echo "===================================================="
        echo "✅ 验证成功！Gemini 已准备就绪。"
        touch "$SUCCESS_MARKER"
        echo "配置已保存。请按 Ctrl+C 停止当前容器，"
        echo "然后运行 'docker compose up -d' 以正常启动代理服务。"
        echo "===================================================="
        exit 0
    else
        echo "===================================================="
        echo "❌ 错误：验证失败！'gemini -p' 无法获得正常响应。"
        echo "可能原因：API Key 无效、配额超限或网络连接问题。"
        echo "已重置初始化状态，下次启动将重新提示。"
        echo "===================================================="
        rm -f "$SUCCESS_MARKER"
        exit 1
    fi
else
    echo "检测到已验证的配置，正在启动 Gemini CLI Proxy..."
    if ! check_gemini; then
        echo "⚠️ 警告：当前配置验证失败，请考虑执行 'docker compose down -v' 重新配置。"
    fi
    
    echo "日志路径: $LOG_FILE"
    exec uv run gemini-cli-proxy --host 0.0.0.0 --port 8765 2>&1 | tee -a "$LOG_FILE"
fi
