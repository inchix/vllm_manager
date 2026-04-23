#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env without sourcing it (no shell execution from the config file).
load_env_file() {
  local file="$1"
  [ -f "$file" ] || return 0
  while IFS= read -r raw || [ -n "$raw" ]; do
    # Strip leading whitespace and optional `export `
    local line="${raw#"${raw%%[![:space:]]*}"}"
    [ -z "$line" ] && continue
    case "$line" in
      '#'*) continue ;;
      export' '*) line="${line#export }" ;;
    esac
    # Must be KEY=... with a valid identifier
    case "$line" in
      [a-zA-Z_]*=*) : ;;
      *) continue ;;
    esac
    local key="${line%%=*}"
    local val="${line#*=}"
    # Validate identifier strictly
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    # Strip surrounding single or double quotes if both ends match
    if [[ "$val" =~ ^\".*\"$ ]]; then
      val="${val:1:${#val}-2}"
    elif [[ "$val" =~ ^\'.*\'$ ]]; then
      val="${val:1:${#val}-2}"
    fi
    # Only set if not already in the environment (env > .env).
    if [ -z "${!key+x}" ]; then
      printf -v "$key" '%s' "$val"
      export "$key"
    fi
  done < "$file"
}

load_env_file "$SCRIPT_DIR/.env"

# Defaults
CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
USE_SUDO="${USE_SUDO:-sudo}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-manager}"
IMAGE_NAME="${IMAGE_NAME:-vllm-manager:latest}"
MODELS_DIR="${MODELS_DIR:-/home/ollama/vllm_models}"
ADMIN_PORT="${ADMIN_PORT:-7080}"
# Host interface the admin UI port is published on. Default to loopback so the
# API is not reachable from the network unless you explicitly opt in.
ADMIN_BIND_HOST="${ADMIN_BIND_HOST:-127.0.0.1}"
VLLM_PORT_START="${VLLM_PORT_START:-8001}"
VLLM_PORT_END="${VLLM_PORT_END:-8010}"
SHM_SIZE="${SHM_SIZE:-16g}"
TMP_SIZE="${TMP_SIZE:-1g}"
CACHE_SIZE="${CACHE_SIZE:-2g}"
GPU_DEVICES="${GPU_DEVICES:-auto}"
READ_ONLY="${READ_ONLY:-true}"
SELINUX_LABEL="${SELINUX_LABEL:-false}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
DETACH="${DETACH:-true}"
AUTH_ENABLED="${AUTH_ENABLED:-true}"
ADMIN_API_KEY="${ADMIN_API_KEY:-}"
HF_TOKEN="${HF_TOKEN:-}"

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
  local -a args=()
  if [ "$GPU_DEVICES" = "auto" ]; then
    for dev in /dev/nvidia[0-9]*; do
      [ -e "$dev" ] && args+=(--device "$dev")
    done
  else
    read -ra GPU_LIST <<< "$GPU_DEVICES"
    for dev in "${GPU_LIST[@]}"; do
      args+=(--device "$dev")
    done
  fi
  for dev in /dev/nvidiactl /dev/nvidia-uvm /dev/nvidia-uvm-tools; do
    [ -e "$dev" ] && args+=(--device "$dev")
  done
  printf '%s\n' "${args[@]}"
}

mapfile -t GPU_ARGS < <(build_gpu_args)

# Volume label for SELinux
VOL_SUFFIX=""
[ "$SELINUX_LABEL" = "true" ] && VOL_SUFFIX=":Z"

# Image prefix for podman (needs localhost/)
IMAGE_REF="$IMAGE_NAME"
if [ "$CONTAINER_RUNTIME" = "podman" ]; then
  case "$IMAGE_NAME" in
    */*) IMAGE_REF="$IMAGE_NAME" ;;
    *)   IMAGE_REF="localhost/$IMAGE_NAME" ;;
  esac
fi

# Build run command
RUN_ARGS=(
  run
  --name "$CONTAINER_NAME"
  --security-opt=label=disable
  --security-opt=no-new-privileges
  --cap-drop=ALL
  --cap-add=SYS_NICE
  --cap-add=IPC_LOCK
  --ulimit nofile=65536:65536
  --pids-limit=-1
  --shm-size="$SHM_SIZE"
  --tmpfs "/tmp:rw,size=$TMP_SIZE"
  --tmpfs "/root/.cache:rw,size=$CACHE_SIZE"
  --tmpfs "/root/.triton:rw,size=$CACHE_SIZE"
  --tmpfs "/root/.config:rw,size=64m"
)

if [ "$DETACH" = "true" ]; then
  RUN_ARGS+=(-d --rm)
else
  RUN_ARGS+=(--rm)
fi

# GPU device flags (already split into individual tokens)
if [ ${#GPU_ARGS[@]} -gt 0 ]; then
  RUN_ARGS+=("${GPU_ARGS[@]}")
fi

[ "$READ_ONLY" = "true" ] && RUN_ARGS+=(--read-only)

# NVIDIA library mounts
[ -n "$NVIDIA_ML_LIB" ] && RUN_ARGS+=(-v "$NVIDIA_ML_LIB:/usr/local/nvidia/lib64/libnvidia-ml.so.1:ro")
[ -n "$CUDA_LIB" ]      && RUN_ARGS+=(-v "$CUDA_LIB:/usr/local/nvidia/lib64/libcuda.so.1:ro")
[ -n "$NVPTX_LIB" ]     && RUN_ARGS+=(-v "$NVPTX_LIB:/usr/local/nvidia/lib64/libnvidia-ptxjitcompiler.so.1:ro")

# Port mappings: bind admin port to the requested host interface; vLLM ports
# stay on 0.0.0.0 because callers usually want LAN-reachable model endpoints.
RUN_ARGS+=(
  -p "$VLLM_PORT_START-$VLLM_PORT_END:$VLLM_PORT_START-$VLLM_PORT_END"
  -p "$ADMIN_BIND_HOST:$ADMIN_PORT:7080"
  -v "$MODELS_DIR:/models${VOL_SUFFIX}"
)

# Pass through auth + HF env vars
RUN_ARGS+=(-e "AUTH_ENABLED=$AUTH_ENABLED")
[ -n "$ADMIN_API_KEY" ] && RUN_ARGS+=(-e "ADMIN_API_KEY=$ADMIN_API_KEY")
[ -n "$HF_TOKEN" ]      && RUN_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
RUN_ARGS+=(-e "VLLM_PORT_START=$VLLM_PORT_START")
RUN_ARGS+=(-e "VLLM_PORT_END=$VLLM_PORT_END")

# Extra args: parsed as a shell word list without eval.
if [ -n "$EXTRA_ARGS" ]; then
  read -ra EXTRA_ARR <<< "$EXTRA_ARGS"
  RUN_ARGS+=("${EXTRA_ARR[@]}")
fi

RUN_ARGS+=("$IMAGE_REF")

echo "Starting $CONTAINER_NAME..."
echo "  Runtime:    $CONTAINER_RUNTIME"
echo "  Admin UI:   http://$ADMIN_BIND_HOST:$ADMIN_PORT"
echo "  Auth:       $([ "$AUTH_ENABLED" = "true" ] && echo enabled || echo DISABLED)"
echo "  vLLM ports: $VLLM_PORT_START-$VLLM_PORT_END"
echo "  Models:     $MODELS_DIR"
echo ""

$CMD "${RUN_ARGS[@]}"

echo "Container started. Open http://$ADMIN_BIND_HOST:$ADMIN_PORT"
