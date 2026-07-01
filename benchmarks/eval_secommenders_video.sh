#!/bin/bash

set -euo pipefail

MODEL_PATH=${1:-}
DATASET=${2:-}
RESULT_NAME=${3:-}
ENABLE_THINKING=${4:-false}
shift $(( $# >= 4 ? 4 : $# ))
EXTRA_ARGS=("$@")

if [ -z "$MODEL_PATH" ] || [ -z "$DATASET" ] || [ -z "$RESULT_NAME" ]; then
    echo "Usage: bash eval_secommenders_video.sh <model_path> <dataset> <result_name> [enable_thinking] [extra evaluate.py args...]"
    echo "Example: bash eval_secommenders_video.sh OpenOneRec/OneRec-1.7B recifvideoxlargeall onerec17b false --sample_size 100"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPENONEREC_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
BENCHMARK_BASE_DIR="${BENCHMARK_BASE_DIR:-${SCRIPT_DIR}}"
VERSION="${VERSION:-v1.0}"

SECOMMENDERS_ALGORITHM_DIR="${SECOMMENDERS_ALGORITHM_DIR:-${OPENONEREC_ROOT}/../Secommenders}"
DATA_DIR="${BENCHMARK_DATA_DIR:-${SECOMMENDERS_ALGORITHM_DIR}/artifacts/openonerec_eval/${DATASET}}"

OUTPUT_DIR="${BENCHMARK_BASE_DIR}/results/${VERSION}/results_${RESULT_NAME}"
LOG_PATH="${BENCHMARK_BASE_DIR}/auto_eval_logs/${VERSION}/${RESULT_NAME}.log"

VIDEO_TEST_PATH="${DATA_DIR}/video/video_test.parquet"
SID2PID_PATH="${DATA_DIR}/sid2pid.json"

show_failure_context() {
    local exit_code=$?
    if [ $exit_code -ne 0 ]; then
        echo
        echo "Evaluation failed with exit code ${exit_code}"
        echo "  log: $LOG_PATH"
        if [ -f "$LOG_PATH" ]; then
            echo "----- log tail -----"
            tail -n 40 "$LOG_PATH" || true
            echo "--------------------"
        fi
    fi
    exit $exit_code
}

trap show_failure_context EXIT

if [ ! -d "$SECOMMENDERS_ALGORITHM_DIR" ]; then
    echo "Secommenders algorithm directory not found: $SECOMMENDERS_ALGORITHM_DIR"
    echo "Set SECOMMENDERS_ALGORITHM_DIR explicitly if your checkout lives elsewhere."
    exit 1
fi

if [ ! -f "$VIDEO_TEST_PATH" ]; then
    echo "Missing exported OpenOneRec video test file: $VIDEO_TEST_PATH"
    echo "Run export_openonerec_video_test.py in secommenders-algorithm first."
    exit 1
fi

if [ ! -f "$SID2PID_PATH" ]; then
    echo "Missing sid2pid.json: $SID2PID_PATH"
    echo "Re-run export_openonerec_video_test.py so the export directory is fully prepared."
    exit 1
fi

if [ -d "$MODEL_PATH" ]; then
    if [ ! -f "$MODEL_PATH/config.json" ]; then
        echo "Model directory does not look like a Hugging Face model root: $MODEL_PATH"
        echo "Expected to find config.json directly under that path."
        echo "You may have passed a parent directory such as ../models/ instead of a concrete model directory."
        exit 1
    fi
fi

mkdir -p "$(dirname "$LOG_PATH")"
mkdir -p "$OUTPUT_DIR"

THINKING_ARGS=()
if [ "$ENABLE_THINKING" = "true" ]; then
    THINKING_ARGS+=(--enable_thinking)
fi

{
    echo "========== Secommenders Video Evaluation =========="
    echo "MODEL_PATH: $MODEL_PATH"
    echo "DATASET: $DATASET"
    echo "DATA_DIR: $DATA_DIR"
    echo "OUTPUT_DIR: $OUTPUT_DIR"
    echo "ENABLE_THINKING: $ENABLE_THINKING"
    echo "EXTRA_ARGS: ${EXTRA_ARGS[*]:-<none>}"
    echo "=================================================="
} >> "$LOG_PATH"

echo "Running video evaluation on Secommenders export"
echo "  model_path: $MODEL_PATH"
echo "  dataset: $DATASET"
echo "  data_dir: $DATA_DIR"
echo "  output_dir: $OUTPUT_DIR"
echo "  log: $LOG_PATH"

PYTHONPATH="${BENCHMARK_BASE_DIR}:${PYTHONPATH:-}" python3 -u scripts/ray-vllm/evaluate.py \
    --task_types video \
    --gpu_memory_utilization 0.8 \
    --model_path "$MODEL_PATH" \
    --data_dir "$DATA_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --dtype bfloat16 \
    --worker_batch_size 256 \
    --overwrite \
    --num_beams 32 \
    --num_return_sequences 32 \
    --num_return_thinking_sequences 1 \
    "${THINKING_ARGS[@]}" \
    "${EXTRA_ARGS[@]}" >> "$LOG_PATH" 2>&1

echo "Evaluation completed successfully"
echo "  results: $OUTPUT_DIR"
echo "  log: $LOG_PATH"
