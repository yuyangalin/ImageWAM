#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../common.sh
source "${SCRIPT_DIR}/../common.sh"
imagewam_init "${SCRIPT_DIR}/../.."
ROBOTWIN_DIR="${ROBOTWIN_DIR:-${REPO_ROOT}/third_party/RoboTwin}"
source ${REPO_ROOT}/.venv/bin/activate

############# USE THIS IF YOUR GPU IS IN DOCKER ENV AND NOT WITH NVIDIA_DRIVER_CAPABILITIES ENV SET TO GRAPHICS

# IMAGEWAM_TMP_DIR="${IMAGEWAM_TMP_DIR:-${REPO_ROOT}/.tmp}"
# FAKE_KMOD_DIR="${FAKE_KMOD_DIR:-${IMAGEWAM_TMP_DIR}/fake-kmod}"
# NVIDIA_RUNFILE="${NVIDIA_RUNFILE:-}"
# VULKAN_ICD_PATH="${VULKAN_ICD_PATH:-${IMAGEWAM_TMP_DIR}/nvidia_icd.json}"


# mkdir -p "${FAKE_KMOD_DIR}" "$(dirname "${VULKAN_ICD_PATH}")"
# for cmd in modprobe rmmod insmod lsmod depmod; do
#   cat > "${FAKE_KMOD_DIR}/${cmd}" <<'EOF'
# #!/usr/bin/env bash
# echo "[fake kmod] $(basename "$0") $@" >&2
# exit 0
# EOF
#   chmod +x "${FAKE_KMOD_DIR}/${cmd}"
# done
# export PATH="${FAKE_KMOD_DIR}:$PATH"

# if [ -n "${NVIDIA_RUNFILE}" ]; then
#   imagewam_run sh "${NVIDIA_RUNFILE}" --accept-license --no-questions --ui=none --no-kernel-module --no-drm --no-nvidia-modprobe
# fi

# USE LINES 
# cat > "${VULKAN_ICD_PATH}" <<'EOF'
# {
#   "file_format_version": "1.0.0",
#   "ICD": {
#     "library_path": "/usr/lib/x86_64-linux-gnu/libGLX_nvidia.so.0",
#     "api_version": "1.3.0"
#   }
# }
# EOF
# export VK_ICD_FILENAMES="${VULKAN_ICD_PATH}"
# export LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:${LD_LIBRARY_PATH:-}"

# imagewam_run apt-get install -y libegl1 libglvnd0 libglx0 libopengl0
# if command -v vulkaninfo >/dev/null 2>&1; then
#   imagewam_run vulkaninfo --summary
# fi

imagewam_require_env ROBOTWIN_DIR
# install environment.
(cd "${ROBOTWIN_DIR}" && imagewam_run bash script/_install_uv.sh)
# prepare assets.
(cd "${ROBOTWIN_DIR}" && imagewam_run bash script/_download_assets.sh)
