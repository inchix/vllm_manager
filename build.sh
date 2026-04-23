#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

load_env_file() {
  local file="$1"
  [ -f "$file" ] || return 0
  while IFS= read -r raw || [ -n "$raw" ]; do
    local line="${raw#"${raw%%[![:space:]]*}"}"
    [ -z "$line" ] && continue
    case "$line" in
      '#'*) continue ;;
      export' '*) line="${line#export }" ;;
    esac
    case "$line" in
      [a-zA-Z_]*=*) : ;;
      *) continue ;;
    esac
    local key="${line%%=*}"
    local val="${line#*=}"
    [[ "$key" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]] || continue
    if [[ "$val" =~ ^\".*\"$ ]]; then
      val="${val:1:${#val}-2}"
    elif [[ "$val" =~ ^\'.*\'$ ]]; then
      val="${val:1:${#val}-2}"
    fi
    if [ -z "${!key+x}" ]; then
      printf -v "$key" '%s' "$val"
      export "$key"
    fi
  done < "$file"
}

load_env_file "$SCRIPT_DIR/.env"

CONTAINER_RUNTIME="${CONTAINER_RUNTIME:-podman}"
USE_SUDO="${USE_SUDO:-sudo}"
IMAGE_NAME="${IMAGE_NAME:-vllm-manager:latest}"

CMD="${USE_SUDO:+$USE_SUDO }${CONTAINER_RUNTIME}"

echo "Building $IMAGE_NAME with $CONTAINER_RUNTIME..."
$CMD build -t "$IMAGE_NAME" "$SCRIPT_DIR"
echo "Done."
