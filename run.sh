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

# --------- Cluster / multi-node (remote GPU over RDMA) ---------
# When CLUSTER_MODE=true the container uses host networking + RDMA passthrough
# so a Ray cluster can span multiple boxes. Run the manager on one box and one
# or more workers (ADMIN_ROLE=worker, CLUSTER_MODE=true, RAY_HEAD_HOST set) on
# the extra GPU boxes. See README "Multi-node / remote GPUs".
CLUSTER_MODE="${CLUSTER_MODE:-false}"
ADMIN_ROLE="${ADMIN_ROLE:-manager}"        # manager | worker
RAY_HEAD_HOST="${RAY_HEAD_HOST:-}"         # manager box IP (required on workers)

# v0.4.0 control plane (opt-in). When true, the container runs the agent/CCP path
# (entrypoint.sh) and needs the SAME host-net + RDMA passthrough as CLUSTER_MODE.
CONTROL_PLANE="${CONTROL_PLANE:-false}"
ROLES="${ROLES:-}"                         # admin,participant,storage (else mapped from ADMIN_ROLE)
CLUSTER_ID="${CLUSTER_ID:-default}"
CCP_ADMIN_URL="${CCP_ADMIN_URL:-}"         # ws(s)://admin:port (workers); derived if empty
CCP_TLS_INSECURE="${CCP_TLS_INSECURE:-}"
NODE_ID="${NODE_ID:-}"
CANONICAL_MODEL_PATH="${CANONICAL_MODEL_PATH:-}"
CCP_HEARTBEAT_SEC="${CCP_HEARTBEAT_SEC:-}"
CCP_TELEMETRY_SEC="${CCP_TELEMETRY_SEC:-}"
CCP_HEARTBEAT_MISS="${CCP_HEARTBEAT_MISS:-}"
RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_NODE_IP="${RAY_NODE_IP:-}"             # this node's IP for Ray control traffic
# NCCL tuning — ALL empty by default (let NCCL auto-detect); set per-fabric in .env.
# Only passed to the container when non-empty, so a stock deploy stays generic.
#   NCCL_IB_HCA        e.g. "mlx4_0:1" — pin to one HCA:port when a card has
#                      multiple ports on different subnets (else NCCL cross-pairs
#                      them and the RDMA QP transition fails).
#   NCCL_IB_GID_INDEX  e.g. "3" for RoCEv2 on some fabrics; empty = NCCL default.
#   NCCL_P2P_DISABLE   set "1" if GPU PCIe P2P is broken (no NVLink + host IOMMU),
#                      which otherwise hangs NCCL at comm init (forces SHM).
NCCL_IB_HCA="${NCCL_IB_HCA:-}"
NCCL_IB_GID_INDEX="${NCCL_IB_GID_INDEX:-}"
NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-}"
# Manager behaviour for multi-GPU instances (read by admin/vllm_manager.py):
#   MULTIGPU_EXECUTOR         single-node multi-GPU executor: mp | ray (default mp)
#   DISABLE_CUSTOM_ALL_REDUCE "true" to add --disable-custom-all-reduce (needed
#                             when GPU P2P is broken; custom all-reduce uses P2P)
MULTIGPU_EXECUTOR="${MULTIGPU_EXECUTOR:-}"
DISABLE_CUSTOM_ALL_REDUCE="${DISABLE_CUSTOM_ALL_REDUCE:-}"
# Heterogeneous NIC names: NICs are named differently per box (enp196s0 on the
# manager, ens2 on ebola). NCCL accepts a COMMA LIST and picks whichever exists
# on each node, and Ray copies the driver's NCCL_SOCKET_IFNAME to all workers —
# so use the SAME list everywhere, but put THIS box's NIC FIRST.
#   manager .env: NCCL_SOCKET_IFNAME=enp196s0,ens2
#   ebola   .env: NCCL_SOCKET_IFNAME=ens2,enp196s0
NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-}"
NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-0}"
# Gloo (vLLM's CPU-side coordination) does NOT accept a comma list — it needs one
# interface that exists on THIS box. Default it to the first entry of the NCCL
# list (this box's NIC, per the ordering rule above); override if needed. Gloo is
# NOT copied to workers by Ray, so each node keeps its own value.
GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-${NCCL_SOCKET_IFNAME%%,*}}"
RDMA_DEVICES="${RDMA_DEVICES:-}"           # space-separated char devices; auto if empty

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

# Models volume is needed on EVERY node — workers load the model locally too,
# so the same files must exist at the same path on each box.
RUN_ARGS+=(-v "$MODELS_DIR:/models${VOL_SUFFIX}")

if [ "$CLUSTER_MODE" = "true" ] || [ "$CONTROL_PLANE" = "true" ]; then
  # Host networking so Ray + NCCL can reach each other on the real host interfaces.
  # A bridged NAT with fixed port maps cannot form a cluster (Ray uses a wide dynamic
  # port range; NCCL needs the RoCE NIC IPs directly). Applies to the legacy
  # CLUSTER_MODE and the v0.4.0 control plane alike.
  RUN_ARGS+=(--network=host)

  # RDMA verbs device passthrough (RoCE/IB). Auto-detect if RDMA_DEVICES unset.
  if [ -z "$RDMA_DEVICES" ]; then
    for d in /dev/infiniband/uverbs* /dev/infiniband/rdma_cm; do
      [ -e "$d" ] && RUN_ARGS+=(--device "$d")
    done
  else
    for d in $RDMA_DEVICES; do
      [ -e "$d" ] && RUN_ARGS+=(--device "$d")
    done
  fi

  # NCCL over the fabric — only pass what's set (empty = NCCL auto-detects).
  [ -n "$NCCL_IB_HCA" ]        && RUN_ARGS+=(-e "NCCL_IB_HCA=$NCCL_IB_HCA")
  [ -n "$NCCL_IB_GID_INDEX" ]  && RUN_ARGS+=(-e "NCCL_IB_GID_INDEX=$NCCL_IB_GID_INDEX")
  RUN_ARGS+=(-e "NCCL_IB_DISABLE=$NCCL_IB_DISABLE")
  [ -n "$NCCL_SOCKET_IFNAME" ] && RUN_ARGS+=(-e "NCCL_SOCKET_IFNAME=$NCCL_SOCKET_IFNAME")
  [ -n "$GLOO_SOCKET_IFNAME" ] && RUN_ARGS+=(-e "GLOO_SOCKET_IFNAME=$GLOO_SOCKET_IFNAME")

  if [ "$CLUSTER_MODE" = "true" ]; then
    # Legacy v0.3.0 Ray wiring consumed by entrypoint.sh.
    RUN_ARGS+=(-e "CLUSTER_MODE=true")
    RUN_ARGS+=(-e "ADMIN_ROLE=$ADMIN_ROLE")
    RUN_ARGS+=(-e "RAY_HEAD_PORT=$RAY_HEAD_PORT")
    [ -n "$RAY_HEAD_HOST" ] && RUN_ARGS+=(-e "RAY_HEAD_HOST=$RAY_HEAD_HOST")
    [ -n "$RAY_NODE_IP" ]   && RUN_ARGS+=(-e "RAY_NODE_IP=$RAY_NODE_IP")
    [ "$ADMIN_ROLE" = "worker" ] && RUN_ARGS+=(--no-healthcheck)
  fi

  if [ "$CONTROL_PLANE" = "true" ]; then
    # v0.4.0 control-plane wiring consumed by entrypoint.sh + the agent.
    RUN_ARGS+=(-e "CONTROL_PLANE=true")
    [ -n "$ROLES" ]                && RUN_ARGS+=(-e "ROLES=$ROLES")
    RUN_ARGS+=(-e "CLUSTER_ID=$CLUSTER_ID")
    RUN_ARGS+=(-e "ADMIN_ROLE=$ADMIN_ROLE")          # legacy fallback for role mapping
    RUN_ARGS+=(-e "RAY_HEAD_PORT=$RAY_HEAD_PORT")
    [ -n "$RAY_HEAD_HOST" ]        && RUN_ARGS+=(-e "RAY_HEAD_HOST=$RAY_HEAD_HOST")
    [ -n "$RAY_NODE_IP" ]          && RUN_ARGS+=(-e "RAY_NODE_IP=$RAY_NODE_IP")
    [ -n "$CCP_ADMIN_URL" ]        && RUN_ARGS+=(-e "CCP_ADMIN_URL=$CCP_ADMIN_URL")
    [ -n "$CCP_TLS_INSECURE" ]     && RUN_ARGS+=(-e "CCP_TLS_INSECURE=$CCP_TLS_INSECURE")
    [ -n "$NODE_ID" ]              && RUN_ARGS+=(-e "NODE_ID=$NODE_ID")
    [ -n "$CANONICAL_MODEL_PATH" ] && RUN_ARGS+=(-e "CANONICAL_MODEL_PATH=$CANONICAL_MODEL_PATH")
    [ -n "$CCP_HEARTBEAT_SEC" ]    && RUN_ARGS+=(-e "CCP_HEARTBEAT_SEC=$CCP_HEARTBEAT_SEC")
    [ -n "$CCP_TELEMETRY_SEC" ]    && RUN_ARGS+=(-e "CCP_TELEMETRY_SEC=$CCP_TELEMETRY_SEC")
    [ -n "$CCP_HEARTBEAT_MISS" ]   && RUN_ARGS+=(-e "CCP_HEARTBEAT_MISS=$CCP_HEARTBEAT_MISS")
    # non-admin nodes have no UI → skip the 7080 healthcheck
    case ",${ROLES}," in
      *,admin,*) : ;;
      ,,) [ "$ADMIN_ROLE" = "worker" ] && RUN_ARGS+=(--no-healthcheck) ;;
      *)  RUN_ARGS+=(--no-healthcheck) ;;
    esac
  fi
else
  # Single-node: publish admin + vLLM ports as before. Admin port binds to the
  # requested host interface; vLLM ports stay on 0.0.0.0 for LAN reachability.
  RUN_ARGS+=(
    -p "$VLLM_PORT_START-$VLLM_PORT_END:$VLLM_PORT_START-$VLLM_PORT_END"
    -p "$ADMIN_BIND_HOST:$ADMIN_PORT:7080"
  )
fi

# Pass through auth + HF env vars
RUN_ARGS+=(-e "AUTH_ENABLED=$AUTH_ENABLED")
[ -n "$ADMIN_API_KEY" ] && RUN_ARGS+=(-e "ADMIN_API_KEY=$ADMIN_API_KEY")
[ -n "$HF_TOKEN" ]      && RUN_ARGS+=(-e "HF_TOKEN=$HF_TOKEN")
RUN_ARGS+=(-e "ADMIN_VLLM_PORT_START=$VLLM_PORT_START")
RUN_ARGS+=(-e "ADMIN_VLLM_PORT_END=$VLLM_PORT_END")

# Persist the torch.compile / Triton / vLLM compile caches on the models volume.
# The rootfs is read-only and /root/.cache + /root/.triton are tmpfs, so without
# this EVERY start recompiles kernels from scratch — pathologically slow on older
# GPUs (e.g. Volta/V100, where some models' torch.compile'd ops take many minutes).
# Each node keeps its own cache (models dirs are per-node). Override CACHE_DIR to
# disable/relocate.
CACHE_DIR="${CACHE_DIR:-/models/.vllm-cache}"
if [ -n "$CACHE_DIR" ]; then
  RUN_ARGS+=(-e "VLLM_CACHE_ROOT=$CACHE_DIR")
  RUN_ARGS+=(-e "TORCHINDUCTOR_CACHE_DIR=$CACHE_DIR/inductor")
  RUN_ARGS+=(-e "TRITON_CACHE_DIR=$CACHE_DIR/triton")
fi

# Hardware/manager tuning — only passed when set (see .env.example for guidance).
[ -n "$NCCL_P2P_DISABLE" ]          && RUN_ARGS+=(-e "NCCL_P2P_DISABLE=$NCCL_P2P_DISABLE")
[ -n "$MULTIGPU_EXECUTOR" ]         && RUN_ARGS+=(-e "MULTIGPU_EXECUTOR=$MULTIGPU_EXECUTOR")
[ -n "$DISABLE_CUSTOM_ALL_REDUCE" ] && RUN_ARGS+=(-e "DISABLE_CUSTOM_ALL_REDUCE=$DISABLE_CUSTOM_ALL_REDUCE")

# Extra args: parsed as a shell word list without eval.
if [ -n "$EXTRA_ARGS" ]; then
  read -ra EXTRA_ARR <<< "$EXTRA_ARGS"
  RUN_ARGS+=("${EXTRA_ARR[@]}")
fi

RUN_ARGS+=("$IMAGE_REF")

echo "Starting $CONTAINER_NAME..."
echo "  Runtime:    $CONTAINER_RUNTIME"
echo "  Role:       $ADMIN_ROLE"
if [ "$CLUSTER_MODE" = "true" ]; then
  echo "  Cluster:    ENABLED (host network, RDMA passthrough, Ray)"
  if [ "$ADMIN_ROLE" = "worker" ]; then
    echo "  Ray head:   ${RAY_HEAD_HOST:-<unset!>}:$RAY_HEAD_PORT"
  else
    echo "  Ray head:   this box:$RAY_HEAD_PORT"
    echo "  Admin UI:   http://$ADMIN_BIND_HOST:$ADMIN_PORT (host network)"
  fi
  echo "  NCCL:       HCA=$NCCL_IB_HCA GID=$NCCL_IB_GID_INDEX IFNAME=${NCCL_SOCKET_IFNAME:-auto}"
else
  echo "  Admin UI:   http://$ADMIN_BIND_HOST:$ADMIN_PORT"
  echo "  Auth:       $([ "$AUTH_ENABLED" = "true" ] && echo enabled || echo DISABLED)"
  echo "  vLLM ports: $VLLM_PORT_START-$VLLM_PORT_END"
fi
echo "  Models:     $MODELS_DIR"
echo ""

$CMD "${RUN_ARGS[@]}"

echo "Container started. Open http://$ADMIN_BIND_HOST:$ADMIN_PORT"
