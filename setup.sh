#!/bin/bash
# setup.sh — generate a .env for vLLM Manager by auto-detecting the local hardware
# (GPUs, NVLink, IOMMU, RDMA HCA/port/GID, RoCE NICs) and prompting for the few
# things that can't be detected (role, head host, models dir). Nothing about any
# specific fabric is hardcoded — it's all discovered or asked for.
#
# Usage:
#   ./setup.sh                                  # interactive
#   ./setup.sh --role manager                   # manager node (admin UI + Ray head)
#   ./setup.sh --role worker --head-host <IP>   # extra GPU box joining the cluster
#   ./setup.sh --role single                    # single-node, no cluster
#   ./setup.sh --yes ...                         # non-interactive (accept detections)
# Flags: --models-dir P --node-ip IP --roce-nic IFACE --api-key K --runtime podman|docker
#        --no-build --no-run --force (overwrite existing .env)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_FILE="$SCRIPT_DIR/.env"

ROLE="" HEAD_HOST="" MODELS_DIR="" NODE_IP="" API_KEY="" ROCE_NIC="" RUNTIME=""
ASSUME_YES=0 DO_BUILD=ask DO_RUN=ask FORCE=0
while [ $# -gt 0 ]; do case "$1" in
  --role) ROLE="$2"; shift 2;;
  --head-host) HEAD_HOST="$2"; shift 2;;
  --models-dir) MODELS_DIR="$2"; shift 2;;
  --node-ip) NODE_IP="$2"; shift 2;;
  --roce-nic) ROCE_NIC="$2"; shift 2;;
  --api-key) API_KEY="$2"; shift 2;;
  --runtime) RUNTIME="$2"; shift 2;;
  --yes|-y) ASSUME_YES=1; shift;;
  --no-build) DO_BUILD=0; shift;;
  --no-run) DO_RUN=0; shift;;
  --force) FORCE=1; shift;;
  -h|--help) sed -n '2,20p' "$0"; exit 0;;
  *) echo "unknown arg: $1" >&2; exit 1;;
esac; done

info(){ printf '  %s\n' "$*"; }
ask(){ # ask VAR "prompt" "default"
  local __v="$1" prompt="$2" def="${3:-}"; local cur="${!__v}"
  [ -n "$cur" ] && return 0
  if [ "$ASSUME_YES" = 1 ] || [ ! -t 0 ]; then printf -v "$__v" '%s' "$def"; return 0; fi
  local ans; read -r -p "$prompt${def:+ [$def]}: " ans </dev/tty || ans=""
  printf -v "$__v" '%s' "${ans:-$def}"
}

# ---------------- hardware detection (best-effort; empty when unknown) ----------------
det_gpu_count(){ nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l | tr -d ' '; }
det_nvlink(){   # "yes" only if at least one link is active
  local o; o=$(nvidia-smi nvlink --status 2>/dev/null || true)
  { echo "$o" | grep -qiE "Link [0-9]+:" && ! echo "$o" | grep -qi "inactive"; } && echo yes || echo no; }
det_iommu(){ [ -n "$(ls /sys/kernel/iommu_groups 2>/dev/null)" ] && echo yes || echo no; }
det_hca(){ ls /sys/class/infiniband/ 2>/dev/null | head -1; }
det_ports(){ ls "/sys/class/infiniband/$1/ports/" 2>/dev/null | sort -n; }
det_gid(){   # RoCEv2 IPv4 GID index for hca $1 port $2
  local h="$1" p="$2" i t g
  for i in $(ls "/sys/class/infiniband/$h/ports/$p/gids/" 2>/dev/null | sort -n); do
    t=$(cat "/sys/class/infiniband/$h/ports/$p/gid_attrs/types/$i" 2>/dev/null || true)
    g=$(cat "/sys/class/infiniband/$h/ports/$p/gids/$i" 2>/dev/null || true)
    case "$t" in *v2*) case "$g" in *ffff:????:????|*ffff:*)
      [ "$g" != "0000:0000:0000:0000:0000:0000:0000:0000" ] && { echo "$i"; return; };; esac;; esac
  done; }
det_netdev(){ rdma link show 2>/dev/null | sed -n "s#.*$1/$2 .*netdev \([^ ]*\).*#\1#p" | head -1; }
det_ip(){ ip -o -4 addr show dev "$1" 2>/dev/null | awk '{print $4}' | cut -d/ -f1 | head -1; }
det_mgmt_ip(){ ip -o -4 addr show 2>/dev/null | awk '!/ lo /{print $4}' | grep -vE '^127\.' | cut -d/ -f1 | head -1; }

echo "== vLLM Manager setup =="
GPUS=$(det_gpu_count); NVLINK=$(det_nvlink); IOMMU=$(det_iommu)
HCA=$(det_hca); PORTS=""; NPORTS=0; GID=""; RNIC=""; RIP=""
[ -n "$HCA" ] && { PORTS=$(det_ports "$HCA"); NPORTS=$(echo "$PORTS" | grep -c .); }
FIRST_PORT=$(echo "$PORTS" | head -1)
[ -n "$HCA" ] && [ -n "$FIRST_PORT" ] && { GID=$(det_gid "$HCA" "$FIRST_PORT"); RNIC=$(det_netdev "$HCA" "$FIRST_PORT"); }
[ -n "$RNIC" ] && RIP=$(det_ip "$RNIC")
info "GPUs: ${GPUS:-0}   NVLink active: $NVLINK   IOMMU: $IOMMU"
info "RDMA HCA: ${HCA:-none} (${NPORTS} port(s))   RoCEv2 GID idx: ${GID:-?}   RoCE NIC: ${RNIC:-none} ${RIP:+($RIP)}"

# Derived, non-hardcoded defaults:
IB_HCA_DEF=""; [ -n "$HCA" ] && { [ "$NPORTS" -gt 1 ] && IB_HCA_DEF="$HCA:$FIRST_PORT" || IB_HCA_DEF="$HCA"; }
# Broken PCIe P2P (no NVLink + IOMMU translated) → disable P2P + custom all-reduce.
P2P_DEF=0; DCAR_DEF=false
{ [ "$NVLINK" = no ] && [ "$IOMMU" = yes ]; } && { P2P_DEF=1; DCAR_DEF=true; }

# ---------------- prompts (only what can't be detected) ----------------
ask ROLE "Role (manager|worker|single)" "${ROLE:-manager}"
ask RUNTIME "Container runtime (podman|docker)" "${RUNTIME:-podman}"
ask MODELS_DIR "Models directory (shared NFS path recommended for clusters)" "${MODELS_DIR:-/models}"
CLUSTER=false; [ "$ROLE" = manager ] || [ "$ROLE" = worker ] && CLUSTER=true
if [ "$CLUSTER" = true ]; then
  ask NODE_IP "This node's cluster IP (Ray control traffic)" "${NODE_IP:-$(det_mgmt_ip)}"
  [ "$ROLE" = worker ] && ask HEAD_HOST "Manager (Ray head) IP" "$HEAD_HOST"
  ask ROCE_NIC "This node's RoCE NIC for NCCL (first in the list)" "${ROCE_NIC:-$RNIC}"
fi
[ "$ROLE" = manager ] || [ "$ROLE" = single ] && ask API_KEY "Admin API key (blank = auto-generate on first run)" "$API_KEY"

# ---------------- write .env ----------------
if [ -f "$ENV_FILE" ] && [ "$FORCE" != 1 ]; then
  echo "Refusing to overwrite existing $ENV_FILE (use --force)"; exit 1
fi
{
  echo "# Generated by setup.sh on $(hostname) — edit as needed."
  echo "CONTAINER_RUNTIME=$RUNTIME"
  echo "MODELS_DIR=$MODELS_DIR"
  echo "ADMIN_BIND_HOST=0.0.0.0"
  [ -n "$API_KEY" ] && echo "ADMIN_API_KEY=$API_KEY"
  if [ "$CLUSTER" = true ]; then
    echo "CLUSTER_MODE=true"
    echo "ADMIN_ROLE=$ROLE"
    [ "$ROLE" = worker ] && echo "CONTAINER_NAME=vllm-worker"
    [ -n "$NODE_IP" ] && echo "RAY_NODE_IP=$NODE_IP"
    [ "$ROLE" = worker ] && [ -n "$HEAD_HOST" ] && echo "RAY_HEAD_HOST=$HEAD_HOST"
    [ -n "$ROCE_NIC" ] && echo "NCCL_SOCKET_IFNAME=$ROCE_NIC   # append peers' NIC names (comma, local first) if they differ"
  fi
  # Auto-detected NCCL fabric tuning (only emit when detected)
  [ -n "$IB_HCA_DEF" ] && echo "NCCL_IB_HCA=$IB_HCA_DEF"
  [ -n "$GID" ]        && echo "NCCL_IB_GID_INDEX=$GID"
  [ "$P2P_DEF" = 1 ]   && echo "NCCL_P2P_DISABLE=1"
  echo "MULTIGPU_EXECUTOR=mp"
  echo "DISABLE_CUSTOM_ALL_REDUCE=$DCAR_DEF"
} > "$ENV_FILE"
echo "Wrote $ENV_FILE:"; sed 's/^/    /' "$ENV_FILE"

# ---------------- build / run ----------------
if [ "$DO_BUILD" = ask ]; then ask DO_BUILD "Build the image now? (y/n)" y; fi
case "$DO_BUILD" in y|yes|1) bash "$SCRIPT_DIR/build.sh";; esac
if [ "$DO_RUN" = ask ]; then ask DO_RUN "Start the container now? (y/n)" y; fi
case "$DO_RUN" in y|yes|1) bash "$SCRIPT_DIR/run.sh";; esac

echo "Done. Manager/single: open http://$(det_mgmt_ip):${ADMIN_PORT:-7080}"
[ "$ROLE" = worker ] && echo "Worker joined head ${HEAD_HOST:-?}. Verify on the manager's /api/cluster."
