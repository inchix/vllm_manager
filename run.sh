#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

# Defaults
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
USE_SUDO="${USE_SUDO:-sudo}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-manager}"
IMAGE_NAME="${IMAGE_NAME:-vllm-manager:latest}"
MODELS_DIR="${MODELS_DIR:-/home/ollama/vllm_models}"
ADMIN_PORT="${ADMIN_PORT:-7080}"
VLLM_PORT_START="${VLLM_PORT_START:-8001}"
VLLM_PORT_END="${VLLM_PORT_END:-8010}"
SHM_SIZE="${SHM_SIZE:-16g}"
TMP_SIZE="${TMP_SIZE:-1g}"
CACHE_SIZE="${CACHE_SIZE:-2g}"
GPU_DEVICES="${GPU_DEVICES:-auto}"
READ_ONLY="${READ_ONLY:-true}"
SELINUX_LABEL="${SELINUX_LABEL:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"

# Build command prefix
CMD="${USE_SUDO:+$USE_SUDO }${CONTAINER_RUNTIME}"

# Detect NVIDIA driver libraries
find_nvidia_lib() {
  local name="$1"
  local path=""
  path=$(ldconfig -p 2>/dev/null | grep "$name" | head -1 | awk '{print $NF}')
  if [ -z "$path" ]; then
    # Fallback: search common paths
    for dir in /usr/lib64 /usr/lib/x86_64-linux-gnu /usr/lib; do
      if ls "$dir"/$name* >/dev/null 2>&1; then
        path=$(ls "$dir"/$name* 2>/dev/null | head -1)
        break
      fi
    done
  fi
  echo "$path"
}

NVIDIA_ML_LIB="${NVIDIA_ML_LIB:-$(find_nvidia_lib libnvidia-ml.so)}"
CUDA_LIB="${CUDA_LIB:-$(find_nvidia_lib libcuda.so)}"
NVPTX_LIB="${NVPTX_LIB:-$(find_nvidia_lib libnvidia-ptxjitcompiler.so)}"

# Detect GPU devices
build_gpu_args() {
  local args=""
  if [ "$GPU_DEVICES" = "auto" ]; then
    # Add all nvidia GPU devices found
    for dev in /dev/nvidia[0-9]*; do
      [ -e "$dev" ] && args="$args --device $dev"
    done
  else
    for dev in $GPU_DEVICES; do
      args="$args --device $dev"
    done
  fi
  # Always add control devices
  for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [ -e "$dev" ] && args="$args --device $dev"
  done
  echo "$args"
}

GPU_ARGS=$(build_gpu_args)

# Volume label for SELinux
VOL_SUFFIX=""
[ "$SELINUX_LABEL" = "true" ] && VOL_SUFFIX=":Z"

# Image prefix for podman (needs localhost/)
IMAGE_REF="$IMAGE_NAME"
if [ "$CONTAINER_RUNTIME" = "podman" ]; then
  # Add localhost/ prefix if no registry specified
  case "$IMAGE_NAME" in
    */*) IMAGE_REF="$IMAGE_NAME" ;;
    *)   IMAGE_REF="localhost/$IMAGE_NAME" ;;
  esac
fi

# Build run command
RUN_ARGS=(
  run -d
  --rm
  --name "$CONTAINER_NAME"
  --security-opt=label=disable
  --ulimit nofile=65536:65536
  --pids-limit=-1
  $GPU_ARGS
  --shm-size="$SHM_SIZE"
  --tmpfs "/tmp:rw,size=$TMP_SIZE"
  --tmpfs "/root/.cache:rw,size=$CACHE_SIZE"
  --tmpfs "/root/.triton:rw,size=$CACHE_SIZE"
  --tmpfs "/root/.config:rw,size=64m"
)

[ "$READ_ONLY" = "true" ] && RUN_ARGS+=(--read-only)

# NVIDIA library mounts
[ -n "$NVIDIA_ML_LIB" ] && RUN_ARGS+=(-v "$NVIDIA_ML_LIB:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro")
[ -n "$CUDA_LIB" ]      && RUN_ARGS+=(-v "$CUDA_LIB:/usr/local/nvidia/lib64/libcuda.so.1:ro")
[ -n "$NVPTX_LIB" ]     && RUN_ARGS+=(-v "$NVPTX_LIB:/usr/local/nvidia/lib64/libnvidia-ptxjitcompiler.so.1:ro")

# Port mappings
RUN_ARGS+=(
  -p "$VLLM_PORT_START-$VLLM_PORT_END:$VLLM_PORT_START-$VLLM_PORT_END"
  -p "$ADMIN_PORT:7080"
  -v "$MODELS_DIR:/models${VOL_SUFFIX}"
)

# Extra args
[ -n "$EXTRA_ARGS" ] && RUN_ARGS+=($EXTRA_ARGS)

RUN_ARGS+=("$IMAGE_REF")

echo "Starting $CONTAINER_NAME..."
echo "  Runtime:    $CONTAINER_RUNTIME"
echo "  Admin UI:   http://localhost:$ADMIN_PORT"
echo "  vLLM ports: $VLLM_PORT_START-$VLLM_PORT_END"
echo "  Models:     $MODELS_DIR"
echo ""

$CMD "${RUN_ARGS[@]}"

echo "Container started. Open http://localhost:$ADMIN_PORT"
