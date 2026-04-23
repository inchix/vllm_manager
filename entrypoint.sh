#!/bin/bash
set -euo pipefail

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
