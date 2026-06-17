#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."

GPU_PER_NODE="${GPU_PER_NODE:-8}"
TASK_TYPE="${TASK_TYPE:-robotwin}"        # robotwin
PRECOMPUTE_QWEN_CACHE="${PRECOMPUTE_QWEN_CACHE:-false}"

imagewam_require_env DATA_ROOT
imagewam_require_env OMNIGEN2_SRC
imagewam_require_env OMNIGEN2_MODEL_PATH
imagewam_require_env QWEN_MODEL_PATH

case "${TASK_TYPE}" in
  robotwin)
    ACTION_DIM=14
    TASK_NAME="robotwin_omnigen2_cache_idm"
    ROBOTWIN_ROOT="${ROBOTWIN_ROOT:-${DATA_ROOT}/robotwin2.0}"
    QWEN_CACHE_DIR="${QWEN_CACHE_DIR:-${ROBOTWIN_ROOT}/qwen_cache}"
    NONIDLE_FILTER_PATH="${NONIDLE_FILTER_PATH:-${ROBOTWIN_ROOT}/nonidle_ranges.json}"
    DATASET_OVERRIDES=(
      "data.train.dataset_dirs=[${ROBOTWIN_ROOT}]"
      "data.val.dataset_dirs=[${ROBOTWIN_ROOT}]"
      "data.train.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
      "data.val.qwen_text_cache_dir=${QWEN_CACHE_DIR}"
      "data.train.nonidle_filter_path=${NONIDLE_FILTER_PATH}"
      "data.val.nonidle_filter_path=${NONIDLE_FILTER_PATH}"
      "model.proprio_dim=14"
    )
    ;;
  *) echo "Invalid TASK_TYPE=${TASK_TYPE}; cache IDM currently expects robotwin" >&2; exit 1 ;;
esac

ACTION_INIT="${ACTION_INIT:-checkpoints/action_dit_omnigen2_${TASK_TYPE}_init.pt}"
export PYTHONPATH="${REPO_ROOT}/src:${OMNIGEN2_SRC}${PYTHONPATH:+:${PYTHONPATH}}"

imagewam_print_config TASK_TYPE TASK_NAME DATA_ROOT OMNIGEN2_SRC OMNIGEN2_MODEL_PATH QWEN_MODEL_PATH QWEN_CACHE_DIR ACTION_INIT

if [ "${REBUILD_ACTION_INIT:-false}" = "true" ] || [ ! -f "${ACTION_INIT}" ]; then
  imagewam_run imagewam_python scripts/omnigen2/preprocess_action_dit_omnigen2.py \
    --model-config configs/model/imagewam_omnigen2.yaml \
    --omnigen2-model-path "${OMNIGEN2_MODEL_PATH}" \
    --action-dim "${ACTION_DIM}" \
    --output "${ACTION_INIT}"
fi

if [ "${PRECOMPUTE_QWEN_CACHE}" = "true" ]; then
  imagewam_run torchrun --standalone --nproc_per_node="${GPU_PER_NODE}" \
    scripts/omnigen2/precompute_qwen_embeds.py \
    task="${TASK_NAME}" \
    model.qwen_path="${QWEN_MODEL_PATH}" \
    qwen_cache_batch_size="${QWEN_CACHE_BATCH_SIZE:-32}" \
    qwen_cache_save_workers="${QWEN_CACHE_SAVE_WORKERS:-4}" \
    qwen_cache_overwrite="${QWEN_CACHE_OVERWRITE:-false}" \
    "${DATASET_OVERRIDES[@]}"
fi

TASK="${TASK_NAME}" imagewam_run bash scripts/omnigen2/train_omnigen2_imagewam.sh "${GPU_PER_NODE}" \
  model.omnigen2_model_path="${OMNIGEN2_MODEL_PATH}" \
  model.omnigen2_vae_path="${OMNIGEN2_MODEL_PATH}" \
  model.qwen_path="${QWEN_MODEL_PATH}" \
  model.action_dit_pretrained_path="${ACTION_INIT}" \
  model.pack_proprio_after_text=true \
  batch_size=4 \
  gradient_accumulation_steps=6 \
  "${DATASET_OVERRIDES[@]}" \
  "$@"
