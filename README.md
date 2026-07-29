# vLLM Manager — by dracotel.com

Web-based admin UI for running multiple vLLM instances across multiple GPUs. Manages vLLM processes inside a single container with a FastAPI backend and vanilla HTML/JS frontend.

## Features

- **Multi-instance** — Run multiple vLLM instances simultaneously, each on different GPUs and ports
- **GPU management** — Select GPUs per instance with conflict detection and mixed-architecture warnings
- **Model downloads** — Download models from HuggingFace directly from the UI with progress tracking (supports `HF_TOKEN` for gated repos)
- **Live monitoring** — Real-time status, per-instance logs, GPU memory charts
- **API-key auth** — Admin UI is gated by an API key by default; can be disabled for isolated on-prem networks
- **Auto-restart on crash** — vLLM instances are re-launched with exponential backoff (configurable per instance)
- **Config persistence** — Running instances are restored when the container restarts
- **Hardened container** — Read-only rootfs, `--cap-drop=ALL`, `no-new-privileges`, tmpfs for scratch
- **Docker & Podman** — Works with both container runtimes
- **Configurable** — All settings (ports, volumes, GPUs) via `.env` file (parsed safely, never sourced)

## Quick Start

```bash
git clone https://github.com/inchix/vllm_manager.git
cd vllm_manager

cp .env.example .env
# Edit .env — set MODELS_DIR, adjust ports, choose docker/podman

bash build.sh
bash run.sh

# Container prints the admin API key on first start (unless you set ADMIN_API_KEY).
open http://127.0.0.1:7080
```

## Requirements

- **Container runtime**: Docker 20+ or Podman 4+
- **NVIDIA GPUs** with drivers on the host
- **NVIDIA Container Toolkit** (Docker) or **CDI configuration** (Podman)
- Model files in a host directory (or download via the UI)

## Authentication

On first run with `AUTH_ENABLED=true` (the default) and no `ADMIN_API_KEY` set, the container generates a random 32-byte URL-safe key and prints it to the logs:

```
[auth] No ADMIN_API_KEY set; generated one for this session.
[auth]     Admin API key: abc123...
[auth]     Set ADMIN_API_KEY to keep it stable across restarts.
```

Open the UI, paste the key into the login page, and you're in. The key is stored in a same-site HttpOnly cookie for 30 days.

API callers can authenticate either header:

```bash
curl -H "X-API-Key: $ADMIN_API_KEY" http://127.0.0.1:7080/api/status
# or
curl -H "Authorization: Bearer $ADMIN_API_KEY" http://127.0.0.1:7080/api/status
```

### Running without auth (on-prem / isolated networks)

If the host sits on a trusted internal network and you've decided auth is unnecessary:

```ini
# .env
AUTH_ENABLED=false
```

This disables auth middleware entirely. The startup banner warns about it. Combine with `ADMIN_BIND_HOST=0.0.0.0` if you want the UI reachable across the LAN. Don't do this on any host that's reachable from the public internet.

## Configuration

Copy `.env.example` to `.env` and edit. `.env` is parsed as simple `KEY=VALUE` pairs — it is **not** sourced by the shell, so `$(...)` and backticks in values are treated as literal text.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_RUNTIME` | `podman` | `podman` or `docker` |
| `USE_SUDO` | `sudo` | Set empty to run without sudo |
| `CONTAINER_NAME` | `vllm-manager` | Container name |
| `IMAGE_NAME` | `vllm-manager:latest` | Built image name |
| `DETACH` | `true` | Detach the container from the terminal. systemd sets this to `false`. |
| `MODELS_DIR` | `/home/ollama/vllm_models` | Host path to model files |
| `ADMIN_PORT` | `7080` | Admin UI port on the host |
| `ADMIN_BIND_HOST` | `127.0.0.1` | Host interface the admin port binds to |
| `AUTH_ENABLED` | `true` | Require API key auth for the admin UI |
| `ADMIN_API_KEY` |  | API key (generated if empty) |
| `HF_TOKEN` |  | HuggingFace token for gated model downloads |
| `VLLM_PORT_START` | `8001` | First vLLM API port |
| `VLLM_PORT_END` | `8010` | Last vLLM API port |
| `SHM_SIZE` | `16g` | Shared memory (needed for tensor parallelism) |
| `GPU_DEVICES` | `auto` | GPU devices to pass through (`auto` detects all) |
| `READ_ONLY` | `true` | Read-only container filesystem |
| `SELINUX_LABEL` | `false` | Add `:Z` label for SELinux (RHEL/Fedora) |
| `EXTRA_ARGS` |  | Extra arguments for the container runtime |

### NVIDIA Library Auto-Detection

The run script automatically finds host NVIDIA driver libraries (`libnvidia-ml`, `libcuda`, `libnvidia-ptxjitcompiler`) and mounts them into the container. Override with `NVIDIA_ML_LIB`, `CUDA_LIB`, `NVPTX_LIB` if auto-detection fails.

## Usage

### Admin UI

Open `http://127.0.0.1:7080` (or your configured `ADMIN_PORT`).

**Starting an instance:**
1. Select one or more GPUs (already-used GPUs are disabled)
2. Choose a model from the dropdown
3. Adjust configuration (memory utilization, max context length, dtype, auto-restart)
4. Click **Start Instance**

**Auto-restart**: if enabled, a crashed vLLM process is automatically re-launched with exponential backoff (2s, 4s, 8s, 16s, 32s). After 5 consecutive failures the instance is marked as errored. A clean run of >5 minutes resets the retry counter.

**Config persistence**: instance configurations are stored in `${MODELS_DIR}/.vllm-manager/instances.json` and restored on container restart.

**Downloading a model:**
1. Enter a HuggingFace repo ID (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
2. Click **Download**
3. Progress bar shows download status and speed (gated repos need `HF_TOKEN`)

### API Endpoints

Each vLLM instance exposes an OpenAI-compatible API on its assigned port (no auth — these are the model endpoints). Example:

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"TinyLlama-1.1B-Chat-v1.0","messages":[{"role":"user","content":"Hello!"}]}'
```

The admin API itself (port 7080) requires the API key when auth is enabled:

```bash
curl -H "X-API-Key: $ADMIN_API_KEY" http://127.0.0.1:7080/api/status
```

See `admin/app.py` for the full endpoint list: `/api/gpus`, `/api/models`, `/api/status`, `/api/start`, `/api/stop`, `/api/download`, `/api/chat`, etc.

The admin container exposes an unauthenticated `/healthz` for the container `HEALTHCHECK`.

## Multi-node / remote GPUs (RDMA)

Cluster mode lets a single vLLM instance span GPUs on **more than one box**, joined
into one Ray cluster over an RDMA fabric (RoCE/InfiniBand). Enable it with
`CLUSTER_MODE=true`. When on, `run.sh` switches the container to host networking,
passes through the RDMA verbs devices (`/dev/infiniband/*`), sets the NCCL RoCE
env, and `entrypoint.sh` starts a **Ray head** (manager) or joins one (worker).

### Topology guidance

Inter-node links (e.g. 40 GbE RoCE) are far slower than intra-node NVLink, so:

- **Tensor-parallel *within* each box** (over NVLink), **pipeline-parallel *across*
  boxes** (over RDMA). PP ships far less data across the slow link than TP.
- The UI's **"Use remote GPUs"** toggle derives this automatically: `PP = number of
  nodes`, `TP = GPUs per node`.
- For **uneven per-node GPU memory** (e.g. a 32 GB box + a 16 GB box), a cluster
  instance **auto-computes a memory-weighted PP layer split** — it probes each node's
  total GPU memory over Ray and gives bigger nodes proportionally more layers (a 64 GB
  box gets 2× the layers of a 32 GB box), ordered to match vLLM's PP-stage assignment
  (driver/manager node = stage 0). This lets a larger model fit than an even split
  would (which would OOM the smaller GPUs). Override any time by passing an explicit
  **PP Layer Partition** (one count per node). Note the model checkpoint must still fit
  on each node's disk, and KV-cache is sized to the tightest stage.

### Requirements (every node)

- The **byte-identical image** (the Containerfile pins the base by digest for this
  reason — build once, distribute to every box).
- The **model files at the same `MODELS_DIR` path**.
- NCCL stages RDMA through host memory here. **GPUDirect RDMA is not available on
  ConnectX-3 (mlx4)** — don't bother loading `nvidia_peermem`; it won't enable GDR
  on this NIC. (Host-staged RoCE RDMA is the expected, working path.)
- **Firewall**: Ray uses many ports beyond the GCS port (raylet, object manager,
  dashboard agent, worker range). If the nodes run `firewalld`, a worker will join
  but then get killed with *"marked as dead by the GCS"* because the head can't
  health-check it back. On each worker, trust the head's IPs (mgmt + RDMA):
  ```bash
  for ip in 172.16.69.201 172.16.254.201 172.16.253.201; do
    sudo firewall-cmd --permanent --zone=trusted --add-source=$ip/32
  done
  sudo firewall-cmd --reload
  ```

### Cluster settings

| Variable | Where | Description |
|----------|-------|-------------|
| `CLUSTER_MODE` | all nodes | `true` to enable host net + RDMA + Ray |
| `ADMIN_ROLE` | all nodes | `manager` (UI + Ray head) or `worker` (Ray worker only) |
| `RAY_HEAD_HOST` | workers | Manager box IP (mgmt net) |
| `RAY_HEAD_PORT` | all nodes | Ray GCS port (default `6379`) |
| `RAY_NODE_IP` | multi-homed | This node's IP for Ray control traffic |
| `NCCL_IB_HCA` | all nodes | RDMA HCA (default `mlx4_0`) |
| `NCCL_IB_GID_INDEX` | all nodes | RoCEv2 GID index (default `3` here) |
| `NCCL_SOCKET_IFNAME` | per node | RoCE NIC (`enp196s0` mgr / `ens2` worker) |
| `RDMA_DEVICES` | optional | Char devices to pass; auto-detected if empty |

### Example: manager + one worker

On the **manager** box (`.env`):

```ini
CLUSTER_MODE=true
ADMIN_ROLE=manager
RAY_NODE_IP=172.16.69.201
NCCL_SOCKET_IFNAME=enp196s0,ens2   # this box's NIC first
```

On the **worker** box (same image, its own `.env`):

```ini
CLUSTER_MODE=true
ADMIN_ROLE=worker
CONTAINER_NAME=vllm-worker
RAY_HEAD_HOST=172.16.69.201
RAY_NODE_IP=172.16.69.230
NCCL_SOCKET_IFNAME=ens2,enp196s0   # this box's NIC first
```

> **Heterogeneous NIC names** are the #1 gotcha. The RoCE NIC is named differently
> per box (`enp196s0` vs `ens2`), and Ray copies the driver's `NCCL_SOCKET_IFNAME`
> to every worker — so a single name can't work cluster-wide. Use a **comma list
> with the local NIC first**: NCCL matches whichever exists on each node, and Gloo
> (which can't parse a list) defaults to the first entry, i.e. the local NIC.

### NCCL tuning for this hardware (V100 + ConnectX-3, no NVLink, IOMMU on)

Two settings (defaulted in `run.sh`, and in the committed `.env` files) are required
here — leave them unless your hardware differs:

- **`NCCL_P2P_DISABLE=1`** — these V100s have **no active NVLink** and the host runs
  **AMD-Vi IOMMU in translated mode**, which breaks GPU PCIe P2P. `nvidia-smi topo
  -p2p r` may say "OK", but NCCL will **hang at comm init** trying P2P. Disabling it
  forces the intra-node **SHM** transport (needs a large `/dev/shm` — `SHM_SIZE`,
  default 16g, provides it). Verify: `nvidia-smi nvlink -s` (inactive) and
  `dmesg | grep -i iommu` (translated).
- **`NCCL_IB_HCA=mlx4_0:1`** — ConnectX-3 exposes **two ports on different subnets**
  (`172.16.254.x` and `172.16.253.x`). Unpinned, NCCL pairs port1↔port2 across nodes
  and the RoCE queue-pair transition fails (`ibv_modify_qp` errno 22, cross-subnet
  GID). Pin to one port so all nodes share a subnet.

> ⚠️ **Known open issue:** even with the above, one inter-node NCCL communicator can
> still stall during channel setup (see `TODO.md`). Single-node instances are
> unaffected; the remote-GPU path needs further NCCL tuning on this fabric.

```bash
# manager box
bash run.sh
# worker box (contributes its GPUs and blocks)
bash run.sh
```

Then open the admin UI: the **GPUs** card shows a **Cluster** panel (joined nodes +
total GPUs), and the launch form gains a **"Use remote GPUs"** toggle once ≥2 nodes
have joined. The cluster's GPU total and layout are managed by Ray — you don't pick
remote GPU indices.

> Cluster mode needs Ray to write to its temp dir (`/tmp/ray`, a tmpfs). If Ray
> complains about a read-only filesystem, set `READ_ONLY=false` on that node.

## Docker vs Podman

### Docker

```bash
# .env
CONTAINER_RUNTIME=docker
USE_SUDO=
# If using NVIDIA Container Toolkit:
EXTRA_ARGS=--gpus all
```

### Podman (rootful)

```bash
# .env
CONTAINER_RUNTIME=podman
USE_SUDO=sudo
```

## Running as a systemd service

`vllm-manager.service` in the repo runs the container in foreground mode so systemd can supervise and restart it:

```bash
sudo cp vllm-manager.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now vllm-manager
```

`Restart=on-failure` means the container comes back automatically on crash.

## Updating the vLLM version

The Containerfile pins the base image by **digest** (not the floating `:nightly`
tag) so every node in a cluster runs the exact same vLLM/NCCL build. To update:

```bash
CONTAINER_RUNTIME=podman   # or docker
sudo $CONTAINER_RUNTIME pull docker.io/vllm/vllm-openai:nightly
sudo $CONTAINER_RUNTIME image inspect docker.io/vllm/vllm-openai:nightly --format '{{.Digest}}'
# Replace the sha256 in Containerfile's FROM line with the printed digest, then:
bash build.sh
```

Rebuild on **every** node from the same digest. `transformers` is pinned `<6`; if a
newer nightly requires `transformers>=6`, bump that constraint in the Containerfile.

### ⚠️ GPU architecture / CUDA constraint (V100 and other Volta cards)

The base image is currently pinned to the **stable** `vllm/vllm-openai:latest`
(vLLM 0.17.1, torch 2.10+cu129) **on purpose**: the current `:nightly` ships torch
built against **CUDA 13 (cu130), which dropped Volta (sm_70)**. On a V100 the newer
build fails at kernel launch with:

```
CUDA error: no kernel image is available for execution on the device
torch ... does not include kernels for this GPU ... compute capability (CC) 7.0
```

CUDA-12 builds (cu126/cu128/cu129) still include `sm_70`, so **stay on a CUDA-12
vLLM build for as long as the cluster runs V100s**. Newer vLLM needs Ampere+
(sm_80+). Before pinning any new digest, verify it supports your GPUs:

```bash
sudo podman run --rm --device /dev/nvidia0 --device /dev/nvidiactl --device /dev/nvidia-uvm \
  --entrypoint python3 <image> -c "import torch; print(torch.cuda.get_arch_list())"
# must include 'sm_70' for V100
```

## Architecture

```
Container (vllm-manager:latest)
├── Admin UI (FastAPI + uvicorn) — port 7080, gated by API key
│   ├── Auth middleware              — /login, /api/auth/*
│   ├── /api/gpus                    — GPU detection via pynvml
│   ├── /api/models                  — Scan /models directory
│   ├── /api/status                  — All instance statuses
│   ├── /api/start, /api/stop        — Lifecycle
│   ├── /api/download                — HuggingFace snapshot download
│   └── /healthz                     — Container healthcheck (unauthenticated)
│
└── vLLM subprocesses (ports 8001..8010)
    └── Managed by VllmManager with auto-restart + persisted configs
```

Each vLLM instance runs as a subprocess managed by the admin backend. GPU isolation is via `CUDA_VISIBLE_DEVICES`. Configs persist to `${MODELS_DIR}/.vllm-manager/instances.json` and are reloaded on startup.

## Project Structure

```
├── Containerfile          # Image definition (extends vllm/vllm-openai)
├── .env.example           # Configuration template
├── build.sh               # Build the container image
├── run.sh                 # Start the container (hardened, loopback by default)
├── stop.sh                # Stop the container
├── entrypoint.sh          # Container entrypoint (admin UI, or Ray head/worker in cluster mode)
├── vllm-manager.service   # systemd unit (Restart=on-failure)
├── admin/
│   ├── __init__.py
│   ├── app.py             # FastAPI backend (auth, lifecycle, persistence)
│   ├── auth.py            # API-key middleware
│   ├── persistence.py     # Instance config persistence
│   ├── vllm_manager.py    # vLLM process lifecycle + auto-restart
│   └── static/
│       ├── index.html     # Single-page admin UI
│       └── login.html     # Sign-in page
├── README.md
└── TODO.md
```

## License

MIT
