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
IMAGE_NAME="${IMAGE_NAME:-vllm-manager:latest}"

CMD="${USE_SUDO:+$USE_SUDO }${CONTAINER_RUNTIME}"

echo "Building $IMAGE_NAME with $CONTAINER_RUNTIME..."
$CMD build -t "$IMAGE_NAME" "$SCRIPT_DIR"
echo "Done."
