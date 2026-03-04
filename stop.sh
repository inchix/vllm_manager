#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Load .env if present
if [ -f "$SCRIPT_DIR/.env" ]; then
  set -a
  source "$SCRIPT_DIR/.env"
  set +a
fi

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
USE_SUDO="${USE_SUDO:-sudo}"
CONTAINER_NAME="${CONTAINER_NAME:-vllm-manager}"

CMD="${USE_SUDO:+$USE_SUDO }${CONTAINER_RUNTIME}"

echo "Stopping $CONTAINER_NAME..."
$CMD stop "$CONTAINER_NAME" 2>/dev/null && echo "Stopped." || echo "Container not running."
