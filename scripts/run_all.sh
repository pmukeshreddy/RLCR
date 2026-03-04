#!/bin/bash
# =============================================================================
# RLCR: Full Pipeline — One Command Reproducibility
# =============================================================================
#
# GPU lifecycle:
#   Steps 1-3:  CPU only (data, teams, embeddings)
#   Step 4:     Launch SGLang with reduced memory (0.40) for rollout generation
#   Step 5:     GRPO training — SGLang does fast batched rollouts,
#               local model does forward pass with gradients + backward
#   Kill SGLang, relaunch with full memory (0.85) for fast eval
#   Step 6:     Evaluation via SGLang
#   Step 7:     Scale comparison (SGLang managed internally)
#   Step 8:     Visualization (CPU only, SGLang killed)
#
# Usage:
#   bash scripts/run_all.sh                    # Full pipeline
#   bash scripts/run_all.sh --no-sglang        # Skip SGLang, use local HF inference
#   bash scripts/run_all.sh --baseline-only    # Steps 1-3 only
#   bash scripts/run_all.sh --quick            # Minimal run for testing
#
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

NO_SGLANG=false
BASELINE_ONLY=false
QUICK=false
CONFIG="configs/default.yaml"
SGLANG_PID=""
SGLANG_PORT=30000

for arg in "$@"; do
    case $arg in
        --no-sglang) NO_SGLANG=true ;;
        --baseline-only) BASELINE_ONLY=true ;;
        --quick) QUICK=true ;;
        --config=*) CONFIG="${arg#*=}" ;;
    esac
done

# Kill SGLang by port — works even if we lost the PID (subshell, etc.)
kill_sglang_on_port() {
    local port="${1:-$SGLANG_PORT}"
    local pids
    pids=$(lsof -ti :"$port" 2>/dev/null || true)
    if [ -n "$pids" ]; then
        echo "[*] Killing SGLang processes on port $port (PIDs: $(echo $pids | tr '\n' ' '))..."
        echo "$pids" | xargs kill -9 2>/dev/null || true
        sleep 3
        echo "[✓] GPU memory freed"
    fi
    SGLANG_PID=""
}

cleanup() {
    echo ""
    echo "[*] Cleaning up..."
    kill_sglang_on_port "$SGLANG_PORT"
}
trap cleanup EXIT INT TERM

echo "============================================="
echo " RLCR: Reinforcement Learning for Code Review"
echo "============================================="
echo " Config: $CONFIG"
echo " SGLang: $([ "$NO_SGLANG" = true ] && echo 'disabled' || echo 'enabled (default)')"
echo " Baseline only: $BASELINE_ONLY"
echo " Quick mode: $QUICK"
echo "============================================="
echo ""

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_DIR="results/logs"
mkdir -p "$LOG_DIR"
LOG_FILE="$LOG_DIR/run_${TIMESTAMP}.log"
echo "Logging to: $LOG_FILE"

run_step() {
    local step_num=$1
    local step_name=$2
    local command=$3

    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Step $step_num: $step_name"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date +%H:%M:%S)] Starting..."

    if eval "$command" 2>&1 | tee -a "$LOG_FILE"; then
        echo "[$(date +%H:%M:%S)] ✓ Step $step_num complete"
    else
        echo "[$(date +%H:%M:%S)] ✗ Step $step_num FAILED"
        echo "Check log: $LOG_FILE"
        exit 1
    fi
}

# Launch SGLang directly (NOT through run_step, to keep PID in parent shell)
launch_sglang() {
    local model="${1:-Qwen/Qwen3-4B}"
    local port="${2:-$SGLANG_PORT}"
    local mem_frac="${3:-0.85}"

    echo "[*] Launching SGLang: $model on port $port (mem_fraction=$mem_frac)"
    python -m sglang.launch_server \
        --model-path "$model" \
        --host 127.0.0.1 \
        --port "$port" \
        --mem-fraction-static "$mem_frac" \
        --trust-remote-code \
        --log-level warning \
        > "$LOG_DIR/sglang_${port}.log" 2>&1 &
    SGLANG_PID=$!

    local waited=0
    local max_wait=300
    while ! curl -s "http://127.0.0.1:${port}/health" > /dev/null 2>&1; do
        sleep 5
        waited=$((waited + 5))
        if ! kill -0 "$SGLANG_PID" 2>/dev/null; then
            echo "[✗] SGLang process died. Check $LOG_DIR/sglang_${port}.log"
            SGLANG_PID=""
            return 1
        fi
        if [ $waited -ge $max_wait ]; then
            echo "[✗] SGLang timed out after ${max_wait}s"
            kill "$SGLANG_PID" 2>/dev/null || true
            SGLANG_PID=""
            return 1
        fi
        [ $((waited % 15)) -eq 0 ] && echo "  ... waiting (${waited}s / ${max_wait}s)"
    done

    echo "[✓] SGLang ready (PID $SGLANG_PID, mem_fraction=$mem_frac)"
    return 0
}

# =============================================
# Steps 1-3: CPU only (data + embeddings)
# =============================================

run_step 1 "Download & Process Data" \
    "python scripts/01_download_data.py --config $CONFIG"

run_step 2 "Team Simulation" \
    "python scripts/02_simulate_teams.py --config $CONFIG"

run_step 3 "Embedding Baseline" \
    "python scripts/03_run_baseline.py --config $CONFIG"

if [ "$BASELINE_ONLY" = true ]; then
    echo ""
    echo "============================================="
    echo " Baseline-only run complete!"
    echo "============================================="
    exit 0
fi

# =============================================
# Step 4+5: Launch SGLang for rollout generation, then train
#
# SGLang runs with reduced memory (0.40) so the local training
# model can coexist on the same GPU for the gradient forward pass.
#
# NOTE: launch_sglang and kill_sglang_on_port are called DIRECTLY
# (not through run_step) so SGLANG_PID stays in the parent shell.
# =============================================

TRAIN_ARGS="--config $CONFIG"
SGLANG_MODEL="Qwen/Qwen3-4B"
if [ "$QUICK" = true ]; then
    TRAIN_ARGS="$TRAIN_ARGS --team security --small"
    SGLANG_MODEL="Qwen/Qwen3-1.7B"
fi

if [ "$NO_SGLANG" = false ]; then
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Step 4: Launch SGLang for Rollout Generation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date +%H:%M:%S)] Starting..."
    launch_sglang "$SGLANG_MODEL" "$SGLANG_PORT" 0.40
    echo "[$(date +%H:%M:%S)] ✓ Step 4 complete"
fi

run_step 5 "GRPO Training (SGLang rollouts + local gradient pass)" \
    "python scripts/05_grpo_train.py $TRAIN_ARGS"

# =============================================
# Kill SGLang (0.40 mem), relaunch with full memory (0.85) for eval
# =============================================

EVAL_ARGS="--config $CONFIG"
if [ "$NO_SGLANG" = true ]; then
    EVAL_ARGS="$EVAL_ARGS --no-sglang"
else
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo " Step 4b: Restart SGLang with full memory for evaluation"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "[$(date +%H:%M:%S)] Starting..."
    kill_sglang_on_port "$SGLANG_PORT"
    launch_sglang "$SGLANG_MODEL" "$SGLANG_PORT" 0.85
    echo "[$(date +%H:%M:%S)] ✓ Step 4b complete"
fi

run_step 6 "Evaluation" \
    "python scripts/06_evaluate.py $EVAL_ARGS"

# =============================================
# Step 7: Scale to larger model (skip in quick mode)
# =============================================

if [ "$QUICK" = false ]; then
    SCALE_ARGS="--config $CONFIG"
    if [ "$NO_SGLANG" = true ]; then
        SCALE_ARGS="$SCALE_ARGS --no-sglang"
    fi
    run_step 7 "Scale to Larger Model" \
        "python scripts/07_scale_4b.py $SCALE_ARGS"
fi

# =============================================
# Step 8: Visualizations (CPU only)
# =============================================

kill_sglang_on_port "$SGLANG_PORT"

run_step 8 "Visualizations" \
    "python scripts/08_visualize.py --config $CONFIG"

echo ""
echo "============================================="
echo " RLCR Pipeline Complete!"
echo "============================================="
echo ""
echo " Results:     results/"
echo " Figures:     results/figures/"
echo " Logs:        $LOG_FILE"
echo " Models:      outputs/"
echo ""
echo " Key outputs:"
echo "   • results/figures/cold_start_all_teams_f1.png  (THE killer chart)"
echo "   • results/figures/team_heatmap.png"
echo "   • results/figures/head_to_head_f1.png"
echo "   • results/figures/results_table.md"
echo ""
