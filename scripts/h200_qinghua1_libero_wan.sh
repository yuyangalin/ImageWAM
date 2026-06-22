#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=common.sh
source "${SCRIPT_DIR}/common.sh"
imagewam_init "${SCRIPT_DIR}/.."

MODE="${1:-}"
if [ -z "${MODE}" ]; then
  cat >&2 <<'USAGE'
Usage:
  bash scripts/h200_qinghua1_libero_wan.sh train-smoke [hydra overrides...]
  bash scripts/h200_qinghua1_libero_wan.sh train [hydra overrides...]
  bash scripts/h200_qinghua1_libero_wan.sh eval-libero CKPT_PATH DATASET_STATS_PATH [hydra overrides...]
  bash scripts/h200_qinghua1_libero_wan.sh eval-libero-plus CKPT_PATH DATASET_STATS_PATH [hydra overrides...]

This helper is intentionally specific to h200-qinghua-1. It reuses the
existing read-only LIBERO data, Wan weights, text cache, and LIBERO benchmark
installations on that host.
USAGE
  exit 2
fi
shift

IMAGEWAM_H200_ENV="${IMAGEWAM_H200_ENV:-/data/home/frank/.conda/envs/fastwam}"
IMAGEWAM_FASTWAM_ROOT="${IMAGEWAM_FASTWAM_ROOT:-/data/home/frank/projects/FastWAM}"
IMAGEWAM_LIBERO_DATA_ROOT="${IMAGEWAM_LIBERO_DATA_ROOT:-${IMAGEWAM_FASTWAM_ROOT}/data/libero_mujoco3.3.2}"
IMAGEWAM_LIBERO_TEXT_CACHE_DIR="${IMAGEWAM_LIBERO_TEXT_CACHE_DIR:-${IMAGEWAM_FASTWAM_ROOT}/data/text_embeds_cache/libero}"
IMAGEWAM_WAN_CHECKPOINT_ROOT="${IMAGEWAM_WAN_CHECKPOINT_ROOT:-${IMAGEWAM_FASTWAM_ROOT}/checkpoints}"
IMAGEWAM_ACTION_INIT="${IMAGEWAM_ACTION_INIT:-${IMAGEWAM_WAN_CHECKPOINT_ROOT}/ActionDiT_linear_interp_Wan22_alphascale_1024hdim.pt}"
IMAGEWAM_LIBERO_PROJECT="${IMAGEWAM_LIBERO_PROJECT:-/data/home/frank/projects/LIBERO}"
IMAGEWAM_LIBERO_PLUS_PROJECT="${IMAGEWAM_LIBERO_PLUS_PROJECT:-/data/home/frank/projects/LIBERO-plus}"

export DATA_ROOT="${DATA_ROOT:-${IMAGEWAM_LIBERO_DATA_ROOT}}"
export IMAGEWAM_LIBERO_DATA_ROOT
export IMAGEWAM_LIBERO_TEXT_CACHE_DIR
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH:-${IMAGEWAM_WAN_CHECKPOINT_ROOT}}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD:-true}"
export PYTHON_BIN="${PYTHON_BIN:-${IMAGEWAM_H200_ENV}/bin/python}"
export PATH="${IMAGEWAM_H200_ENV}/bin:${PATH}"
export LD_LIBRARY_PATH="${IMAGEWAM_H200_ENV}/lib:${LD_LIBRARY_PATH:-}"
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_DIR="${WANDB_DIR:-${REPO_ROOT}/wandb}"
export HF_HOME="${HF_HOME:-/data/home/maxliu/.cache/huggingface}"
export TORCH_HOME="${TORCH_HOME:-/data/home/maxliu/.cache/torch}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-/data/tmp/maxliu/triton-cache}"
export TMPDIR="${TMPDIR:-/data/tmp/maxliu/imagewam}"
mkdir -p "${WANDB_DIR}" "${HF_HOME}" "${TORCH_HOME}" "${TRITON_CACHE_DIR}" "${TMPDIR}" "${REPO_ROOT}/.remote"

cat > "${REPO_ROOT}/.remote/sitecustomize.py" <<'PY'
import logging
import os

_orig_file_handler = logging.FileHandler


class _ImageWAMFileHandler(_orig_file_handler):
    def __init__(self, filename, *args, **kwargs):
        if os.fspath(filename) == "/tmp/robosuite.log":
            filename = os.environ.get(
                "ROBOSUITE_LOG_PATH",
                os.path.join(os.environ.get("TMPDIR", "/tmp"), "robosuite.log"),
            )
            os.makedirs(os.path.dirname(os.path.abspath(filename)), exist_ok=True)
        super().__init__(filename, *args, **kwargs)


logging.FileHandler = _ImageWAMFileHandler
PY

write_libero_config() {
  local flavor="$1"
  local project_root pkg_root
  case "${flavor}" in
    libero)
      project_root="${IMAGEWAM_LIBERO_PROJECT}"
      pkg_root="${project_root}/libero/libero"
      ;;
    libero-plus)
      project_root="${IMAGEWAM_LIBERO_PLUS_PROJECT}"
      pkg_root="${project_root}/libero/libero"
      ;;
    *)
      echo "Invalid LIBERO flavor: ${flavor}" >&2
      exit 2
      ;;
  esac

  mkdir -p "${HOME}/.libero"
  cat > "${HOME}/.libero/config.yaml" <<EOF
benchmark_root: ${pkg_root}
bddl_files: ${pkg_root}/bddl_files
init_states: ${pkg_root}/init_files
datasets: ${IMAGEWAM_LIBERO_DATA_ROOT}/datasets
assets: ${pkg_root}/assets
EOF
  printf '%s\n' "${project_root}"
}

write_worker_env() {
  local flavor="$1"
  local project_root
  project_root="$(write_libero_config "${flavor}")"
  local worker_env="${REPO_ROOT}/.remote/worker_${flavor}.sh"
  cat > "${worker_env}" <<EOF
#!/usr/bin/env bash
export PATH="${IMAGEWAM_H200_ENV}/bin:\${PATH}"
export LD_LIBRARY_PATH="${IMAGEWAM_H200_ENV}/lib:\${LD_LIBRARY_PATH:-}"
export PYTHON_BIN="${PYTHON_BIN}"
export DIFFSYNTH_MODEL_BASE_PATH="${DIFFSYNTH_MODEL_BASE_PATH}"
export DIFFSYNTH_SKIP_DOWNLOAD="${DIFFSYNTH_SKIP_DOWNLOAD}"
export HF_HOME="${HF_HOME}"
export TORCH_HOME="${TORCH_HOME}"
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR}"
export TMPDIR="${TMPDIR}"
export ROBOSUITE_LOG_PATH="${TMPDIR}/robosuite.log"
export TOKENIZERS_PARALLELISM=false
export PYTHONPATH="${REPO_ROOT}/.remote:${project_root}:${REPO_ROOT}/src:${REPO_ROOT}:${REPO_ROOT}/experiments/libero:\${PYTHONPATH:-}"
EOF
  chmod +x "${worker_env}"
  printf '%s\n' "${worker_env}"
}

COMMON_TRAIN_OVERRIDES=(
  "task=libero_uncond_2cam224_1e-4"
  "model.model_id=Wan-AI/Wan2.2-TI2V-5B"
  "model.tokenizer_model_id=Wan-AI/Wan2.1-T2V-1.3B"
  "model.redirect_common_files=false"
  "model.action_dit_pretrained_path=${IMAGEWAM_ACTION_INIT}"
  "+data.train.lerobot_v3_video_backend=pyav"
  "data.train.dataset_dirs=[${IMAGEWAM_LIBERO_DATA_ROOT}/libero_spatial_no_noops_lerobot,${IMAGEWAM_LIBERO_DATA_ROOT}/libero_object_no_noops_lerobot,${IMAGEWAM_LIBERO_DATA_ROOT}/libero_goal_no_noops_lerobot,${IMAGEWAM_LIBERO_DATA_ROOT}/libero_10_no_noops_lerobot]"
  "wandb.enabled=false"
)

run_train() {
  local nproc="$1"
  shift
  export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
  imagewam_print_config DATA_ROOT IMAGEWAM_LIBERO_TEXT_CACHE_DIR DIFFSYNTH_MODEL_BASE_PATH IMAGEWAM_ACTION_INIT PYTHON_BIN
  imagewam_run bash scripts/train_zero1.sh "${nproc}" "${COMMON_TRAIN_OVERRIDES[@]}" "$@"
}

run_eval() {
  local flavor="$1"
  local ckpt_path="$2"
  local stats_path="$3"
  shift 3
  local worker_env
  worker_env="$(write_worker_env "${flavor}")"
  export LIBERO_WORKER_ENV_SOURCE="${worker_env}"
  export ROBOSUITE_LOG_PATH="${TMPDIR}/robosuite.log"
  export PYTHONPATH="${REPO_ROOT}/.remote:$(dirname "$(dirname "${worker_env}")")/src:${REPO_ROOT}:${REPO_ROOT}/experiments/libero:${PYTHONPATH:-}"
  case "${flavor}" in
    libero)
      export PYTHONPATH="${IMAGEWAM_LIBERO_PROJECT}:${PYTHONPATH}"
      ;;
    libero-plus)
      export PYTHONPATH="${IMAGEWAM_LIBERO_PLUS_PROJECT}:${PYTHONPATH}"
      ;;
  esac
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
  export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

  imagewam_print_config LIBERO_WORKER_ENV_SOURCE PYTHONPATH MUJOCO_GL PYOPENGL_PLATFORM
  imagewam_run imagewam_python experiments/libero/run_libero_manager.py \
    --config-name sim_libero \
    task=libero_uncond_2cam224_1e-4 \
    ckpt="${ckpt_path}" \
    EVALUATION.dataset_stats_path="${stats_path}" \
    model.model_id=Wan-AI/Wan2.2-TI2V-5B \
    model.tokenizer_model_id=Wan-AI/Wan2.1-T2V-1.3B \
    model.redirect_common_files=false \
    model.action_dit_pretrained_path=null \
    +data.train.lerobot_v3_video_backend=pyav \
    model.skip_dit_load_from_pretrain=true \
    MULTIRUN.num_gpus="${NUM_GPUS:-1}" \
    MULTIRUN.max_tasks_per_gpu="${MAX_TASKS_PER_GPU:-1}" \
    MULTIRUN.task_suite_names="${TASK_SUITE_NAMES:-[libero_10,libero_goal,libero_spatial,libero_object]}" \
    MULTIRUN.task_sample_ratio="${TASK_SAMPLE_RATIO:-null}" \
    MULTIRUN.chunk_size="${CHUNK_SIZE:-1}" \
    EVALUATION.num_trials="${NUM_TRIALS:-1}" \
    EVALUATION.action_horizon="${ACTION_HORIZON:-16}" \
    EVALUATION.replan_steps="${REPLAN_STEPS:-10}" \
    "$@"
}

case "${MODE}" in
  train-smoke)
    RUN_ID="${RUN_ID:-h200q1_wan_libero_smoke_$(date +%Y%m%d_%H%M%S)}"
    export RUN_ID
    run_train "${GPU_PER_NODE:-1}" \
      batch_size="${BATCH_SIZE:-1}" \
      num_workers="${NUM_WORKERS:-2}" \
      persistent_workers=false \
      prefetch_factor=1 \
      max_steps="${MAX_STEPS:-1}" \
      save_every="${SAVE_EVERY:-1}" \
      eval_every="${EVAL_EVERY:-1000000}" \
      log_every="${LOG_EVERY:-1}" \
      keep_latest_state_only=true \
      "$@"
    ;;
  train)
    RUN_ID="${RUN_ID:-h200q1_wan_libero_$(date +%Y%m%d_%H%M%S)}"
    export RUN_ID
    run_train "${GPU_PER_NODE:-8}" "$@"
    ;;
  eval-libero)
    if [ "$#" -lt 2 ]; then
      echo "eval-libero requires CKPT_PATH and DATASET_STATS_PATH" >&2
      exit 2
    fi
    run_eval libero "$@"
    ;;
  eval-libero-plus)
    if [ "$#" -lt 2 ]; then
      echo "eval-libero-plus requires CKPT_PATH and DATASET_STATS_PATH" >&2
      exit 2
    fi
    run_eval libero-plus "$@"
    ;;
  *)
    echo "Invalid mode: ${MODE}" >&2
    exit 2
    ;;
esac
