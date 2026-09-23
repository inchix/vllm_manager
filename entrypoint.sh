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

# ---------------------------------------------------------------------------
# v0.4.0 control plane (opt-in). When CONTROL_PLANE=true, node behaviour is
# driven by the agent + CCP, not by the legacy Ray-head/worker startup below:
#   admin node        — run uvicorn (which hosts the CCP hub) + a LOCAL agent
#                        (for this box's own participant/storage roles).
#   participant/storage — run the agent only; it dials the admin and Ray is
#                        started per-replica by the agent's runner.
# ---------------------------------------------------------------------------
CONTROL_PLANE="${CONTROL_PLANE:-false}"
if [ "$CONTROL_PLANE" = "true" ]; then
  ROLES_ENV="${ROLES:-}"
  if [ -z "$ROLES_ENV" ]; then
    case "$ROLE" in
      manager|single) ROLES_ENV="admin,participant,storage" ;;
      worker) ROLES_ENV="participant" ;;
    esac
  fi
  export ROLES="$ROLES_ENV"
  case ",$ROLES_ENV," in *,admin,*) IS_ADMIN=1 ;; *) IS_ADMIN=0 ;; esac
  CP_BIND="${ADMIN_CONTAINER_BIND:-0.0.0.0}"
  CP_PORT="${ADMIN_CONTAINER_PORT:-7080}"
  CP_LOG="${LOG_LEVEL:-info}"
  if [ "$IS_ADMIN" = "1" ]; then
    echo "[entrypoint] control plane ON (admin): uvicorn + local agent; roles=$ROLES_ENV"
    : "${CCP_ADMIN_URL:=ws://127.0.0.1:${CP_PORT}}"; export CCP_ADMIN_URL
    # local agent for this box's own roles (retries until uvicorn is up)
    ( sleep 3; exec python3 -m admin.agent.agentd ) &
    exec uvicorn admin.app:app --host "$CP_BIND" --port "$CP_PORT" \
      --log-level "$CP_LOG" --no-access-log
  else
    echo "[entrypoint] control plane ON (worker): agent only; roles=$ROLES_ENV"
    exec python3 -m admin.agent.agentd
  fi
fi

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
