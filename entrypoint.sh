#!/bin/bash
set -euo pipefail

# ---------------------------------------------------------------------------
# Roles:
#   manager (default) — runs the FastAPI admin UI. In cluster mode it first
#                       starts a Ray head so remote worker nodes can join and
#                       vLLM instances can span multiple boxes.
#   worker            — joins an existing Ray head and blocks. No admin UI.
#                       Run this image with ADMIN_ROLE=worker on the extra
#                       GPU box(es); it contributes its GPUs to the cluster.
#
# Cluster mode is gated by CLUSTER_MODE=true. When off, behaviour is identical
# to the original single-node admin server.
# ---------------------------------------------------------------------------

ROLE="${ADMIN_ROLE:-manager}"
CLUSTER_MODE="${CLUSTER_MODE:-false}"

RAY_HEAD_PORT="${RAY_HEAD_PORT:-6379}"
RAY_TMPDIR="${RAY_TMPDIR:-/tmp/ray}"

# Optional explicit node IP for Ray to advertise. Leave empty to let Ray
# auto-detect. On multi-homed boxes (mgmt + RoCE NICs) set this to the address
# the other nodes should use for Ray control traffic (usually the mgmt IP).
RAY_NODE_IP="${RAY_NODE_IP:-}"

ray_node_ip_arg=()
[ -n "$RAY_NODE_IP" ] && ray_node_ip_arg=(--node-ip-address "$RAY_NODE_IP")

start_ray_head() {
  echo "[entrypoint] Starting Ray head on port ${RAY_HEAD_PORT} (temp-dir ${RAY_TMPDIR})"
  ray start --head \
    --port "$RAY_HEAD_PORT" \
    --temp-dir "$RAY_TMPDIR" \
    --disable-usage-stats \
    "${ray_node_ip_arg[@]}"
}

case "$ROLE" in
  worker)
    if [ "$CLUSTER_MODE" != "true" ]; then
      echo "[entrypoint] ADMIN_ROLE=worker requires CLUSTER_MODE=true" >&2
      exit 1
    fi
    : "${RAY_HEAD_HOST:?RAY_HEAD_HOST must be set for a worker (the manager box IP)}"
    echo "[entrypoint] Joining Ray head at ${RAY_HEAD_HOST}:${RAY_HEAD_PORT} as worker"
    # --block keeps the container alive as the Ray worker; it contributes its
    # GPUs to the cluster and does nothing else.
    exec ray start \
      --address "${RAY_HEAD_HOST}:${RAY_HEAD_PORT}" \
      --temp-dir "$RAY_TMPDIR" \
      --disable-usage-stats \
      "${ray_node_ip_arg[@]}" \
      --block
    ;;

  manager)
    if [ "$CLUSTER_MODE" = "true" ]; then
      start_ray_head
    fi
    ;;

  *)
    echo "[entrypoint] Unknown ADMIN_ROLE: $ROLE (expected manager|worker)" >&2
    exit 1
    ;;
esac

# The admin UI binds to 0.0.0.0 inside the container because the host port
# mapping is what controls external reachability (see run.sh ADMIN_BIND_HOST).
BIND="${ADMIN_CONTAINER_BIND:-0.0.0.0}"
PORT="${ADMIN_CONTAINER_PORT:-7080}"
LOG_LEVEL="${LOG_LEVEL:-info}"

exec uvicorn admin.app:app \
  --host "$BIND" \
  --port "$PORT" \
  --log-level "$LOG_LEVEL" \
  --no-access-log
