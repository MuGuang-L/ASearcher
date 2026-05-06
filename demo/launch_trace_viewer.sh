#!/bin/bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

DEFAULT_TRACE_DIR="/tmp/areal/experiments/logs/root/asearcher-light-qwen3/run1/generated"
TRACE_DIR="${1:-$DEFAULT_TRACE_DIR}"
HOST="${2:-127.0.0.1}"
PORT="${3:-8765}"

echo "🚀 启动 ASearcher Trace Viewer"
echo "=================================================="
echo "Trace 目录: $TRACE_DIR"
echo "访问地址: http://$HOST:$PORT"
echo "=================================================="

if ! command -v python3 >/dev/null 2>&1; then
    echo "❌ 错误: 未找到 python3"
    exit 1
fi

if ! python3 -c "import fastapi, uvicorn" 2>/dev/null; then
    echo "❌ 缺少依赖包: fastapi uvicorn"
    echo "请运行: pip install fastapi uvicorn"
    exit 1
fi

if [[ ! -d "$TRACE_DIR" ]]; then
    echo "⚠️  警告: trace 目录不存在，页面可以启动，但暂时不会显示 episode"
fi

cd "${ROOT_DIR}/demo"
python3 light_trace_server.py --trace-dir "$TRACE_DIR" --host "$HOST" --port "$PORT"
