#!/bin/bash
# Step 4: Launch SGLang server via Python subprocess manager.
#
# This is a thin wrapper around src/models/sglang_server.py.
# The Python manager handles:
#   - Killing any existing server on the port
#   - Launching as a managed subprocess
#   - Health-check polling until ready
#   - Graceful shutdown on exit
#
# Usage:
#   bash scripts/04_launch_sglang.sh [MODEL] [PORT]
#   bash scripts/04_launch_sglang.sh Qwen/Qwen3-4B 30000
#
# To stop: kill the Python process, or use SGLangServer.stop() in code.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

MODEL="${1:-Qwen/Qwen3-4B}"
PORT="${2:-30000}"
ENABLE_LORA="${3:-false}"

echo "============================================="
echo " Step 4: SGLang Server"
echo "============================================="
echo " Model: $MODEL"
echo " Port:  $PORT"
echo " LoRA:  $ENABLE_LORA"
echo "============================================="

# Check if already healthy
if curl -s "http://127.0.0.1:${PORT}/health" > /dev/null 2>&1; then
    echo "[✓] SGLang server already running on port $PORT"
    exit 0
fi

# Launch via Python manager
python -c "
import sys
sys.path.insert(0, '.')
from src.models.sglang_server import SGLangServer

server = SGLangServer(
    '${MODEL}',
    port=${PORT},
    enable_lora='${ENABLE_LORA}' == 'true',
    max_lora_rank=32,
)
if server.start():
    print()
    print('[✓] SGLang ready. Server running as subprocess.')
    print(f'    PID: {server.process.pid}')
    print(f'    URL: {server.url}')
    print(f'    To stop: kill {server.process.pid}')
else:
    print('[✗] SGLang failed to start.')
    print('    Check results/logs/sglang_${PORT}.log for details.')
    print()
    print('Troubleshooting:')
    print('  - Check GPU memory: nvidia-smi')
    print('  - Try smaller model: Qwen/Qwen3-1.7B')
    print('  - Run without SGLang: bash scripts/run_all.sh --no-sglang')
    sys.exit(1)
"
