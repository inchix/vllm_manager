# vLLM Manager - by dracotel.com

Web-based admin UI for running multiple vLLM instances across multiple GPUs. Manages vLLM processes inside a single container with a FastAPI backend and vanilla HTML/JS frontend.

## Features

- **Multi-instance** — Run multiple vLLM instances simultaneously, each on different GPUs and ports
- **GPU management** — Select GPUs per instance with conflict detection and mixed-architecture warnings
- **Model downloads** — Download models from HuggingFace directly from the UI with progress tracking
- **Live monitoring** — Real-time status, per-instance logs, GPU memory usage
- **Zero persistence** — Container runs read-only with tmpfs mounts; no state stored in the container
- **Docker & Podman** — Works with both container runtimes
- **Configurable** — All settings (ports, volumes, GPUs) via `.env` file

## Quick Start

```bash
# Clone
git clone https://github.com/inchix/vllm_manager.git
cd vllm_manager

# Configure
cp .env.example .env
# Edit .env — set MODELS_DIR, adjust ports, choose docker/podman

# Build & Run
bash build.sh
bash run.sh

# Open admin UI
open http://localhost:7080
```

## Requirements

- **Container runtime**: Docker 20+ or Podman 4+
- **NVIDIA GPUs** with drivers installed on the host
- **NVIDIA Container Toolkit** (for Docker) or **CDI configuration** (for Podman)
- Model files in a host directory (or download via the UI)

## Configuration

Copy `.env.example` to `.env` and edit. All settings have sensible defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `CONTAINER_RUNTIME` | `podman` | `podman` or `docker` |
| `USE_SUDO` | `sudo` | Set empty to run without sudo |
| `CONTAINER_NAME` | `vllm-manager` | Container name |
| `IMAGE_NAME` | `vllm-manager:latest` | Built image name |
| `MODELS_DIR` | `/home/ollama/vllm_models` | Host path to model files |
| `ADMIN_PORT` | `7080` | Admin UI port |
| `VLLM_PORT_START` | `8001` | First vLLM API port |
| `VLLM_PORT_END` | `8010` | Last vLLM API port |
| `SHM_SIZE` | `16g` | Shared memory (needed for tensor parallelism) |
| `GPU_DEVICES` | `auto` | GPU devices to pass through (`auto` detects all) |
| `READ_ONLY` | `true` | Read-only container filesystem |
| `SELINUX_LABEL` | `false` | Add `:Z` label for SELinux (RHEL/Fedora) |
| `EXTRA_ARGS` | | Extra arguments for the container runtime |

### NVIDIA Library Auto-Detection

The run script automatically finds host NVIDIA driver libraries (`libnvidia-ml`, `libcuda`, `libnvidia-ptxjitcompiler`) and mounts them into the container. Override with `NVIDIA_ML_LIB`, `CUDA_LIB`, `NVPTX_LIB` if auto-detection fails.

## Usage

### Admin UI

Open `http://localhost:7080` (or your configured `ADMIN_PORT`).

**Starting an instance:**
1. Select one or more GPUs (already-used GPUs are disabled)
2. Choose a model from the dropdown
3. Adjust configuration (memory utilization, max context length, dtype)
4. Click **Start Instance**
5. The instance appears in the Instances list with its assigned port

**Downloading a model:**
1. Enter a HuggingFace repo ID (e.g. `TinyLlama/TinyLlama-1.1B-Chat-v1.0`)
2. Click **Download**
3. Progress bar shows download status and speed
4. Model appears in the dropdown when complete

### API Endpoints

Each vLLM instance exposes an OpenAI-compatible API on its assigned port:

```bash
# List models on instance at port 8001
curl http://localhost:8001/v1/models

# Chat completion
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "TinyLlama-1.1B-Chat-v1.0",
    "messages": [{"role": "user", "content": "Hello!"}]
  }'
```

### Admin API

```bash
# List GPUs
curl http://localhost:7080/api/gpus

# List available models
curl http://localhost:7080/api/models

# Get all instance statuses
curl http://localhost:7080/api/status

# Start an instance
curl -X POST http://localhost:7080/api/start \
  -H "Content-Type: application/json" \
  -d '{"model": "TinyLlama-1.1B-Chat-v1.0", "gpu_ids": [0], "dtype": "float16"}'

# Stop an instance
curl -X POST http://localhost:7080/api/stop \
  -H "Content-Type: application/json" \
  -d '{"instance_id": "instance-1"}'

# Download a model
curl -X POST http://localhost:7080/api/download \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "TinyLlama/TinyLlama-1.1B-Chat-v1.0"}'

# Check download progress
curl http://localhost:7080/api/download/status
```

## Docker vs Podman

### Docker

```bash
# .env
CONTAINER_RUNTIME=docker
USE_SUDO=

# If using NVIDIA Container Toolkit, you may want:
EXTRA_ARGS=--gpus all
```

With the NVIDIA Container Toolkit installed, Docker handles GPU passthrough automatically via `--gpus all`. The script's manual device mounting serves as a fallback.

### Podman (rootful)

```bash
# .env
CONTAINER_RUNTIME=podman
USE_SUDO=sudo
```

Podman requires explicit device passthrough. The script auto-detects GPU devices and mounts NVIDIA driver libraries into the container.

## Architecture

```
Container (vllm-manager:latest)
├── Admin UI (FastAPI + uvicorn) — port 7080
│   ├── GET  /           — Web UI
│   ├── GET  /api/gpus   — GPU detection via pynvml
│   ├── GET  /api/models — Scan /models directory
│   ├── GET  /api/status — All instance statuses
│   ├── POST /api/start  — Launch vLLM subprocess
│   ├── POST /api/stop   — Stop vLLM subprocess
│   └── POST /api/download — Download from HuggingFace
│
├── vLLM Instance 1 (subprocess) — port 8001
│   └── OpenAI-compatible API
├── vLLM Instance 2 (subprocess) — port 8002
│   └── OpenAI-compatible API
└── ...up to 10 instances
```

Each vLLM instance runs as a subprocess managed by the admin backend. GPU isolation is achieved via `CUDA_VISIBLE_DEVICES`. Ports are allocated from a pool (8001-8010 by default).

## Project Structure

```
├── Containerfile          # Image definition (extends vllm/vllm-openai)
├── .env.example           # Configuration template
├── build.sh               # Build the container image
├── run.sh                 # Start the container
├── stop.sh                # Stop the container
├── entrypoint.sh          # Container entrypoint (starts admin UI)
├── admin/
│   ├── __init__.py
│   ├── app.py             # FastAPI backend
│   ├── vllm_manager.py    # vLLM process lifecycle manager
│   └── static/
│       └── index.html     # Single-page admin UI
├── README.md
└── TODO.md
```

## License

MIT
