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
├── entrypoint.sh          # Container entrypoint (starts admin UI)
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
