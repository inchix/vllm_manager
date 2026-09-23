# 00 — Overview

## Why v0.4.0

v0.3.0 answered the hard question — *can this hardware do distributed inference at all?* — and
the answer is yes. A dense model runs TP=2 within a box × PP=2 across two boxes, over RoCE
RDMA, with weights served from a shared NFS-over-RDMA volume. All the fabric-level gremlins
(broken PCIe P2P, cross-subnet RoCE QP failures, executor hangs, Volta/CUDA constraints) are
understood and encoded as configuration.

But the *orchestration* around that is still thin, and two independent attempts have now shown
the same seams:

1. **v0.3.0 (`CLUSTER_MODE` / `ADMIN_ROLE=manager|worker`)** runs Ray inside the admin image on
   every box and starts each box by hand (`bash run.sh` here, `bash run.sh` there). There is no
   central view of the cluster, no liveness, and no recovery: when a worker box reboots, the
   whole instance dies and a human must restart the worker and relaunch the model. We lived this
   repeatedly during the ebola power-fault debugging — every reset was a manual recovery.

2. **Community PR #1 (`cluster-two-host-ray`)** reaches for a cleaner split — a CPU-only admin
   that does *not* take GPUs — but implements it by **SSHing into the GPU boxes** to run Docker
   and Ray, and parsing `nvidia-smi` text over the wire. That works, but it means SSH keys on
   the admin, inbound SSH on every worker, imperative remote shell, and text-scraping instead
   of structured state. It is also based on a pre-v0.3.0 `main` and does not merge cleanly.

Both are circling the same missing piece: **a control plane.** v0.4.0 builds it.

## What changes (and what deliberately does not)

**Changes — the control plane is new:**

- Nodes declare a **composable set of roles** (`admin`, `participant`, `storage`) instead of a
  single `ADMIN_ROLE`. One box can be all three; a cluster can split them any way. See
  [01-roles](01-roles.md).
- Every node runs one **agent** that dials the admin over an authenticated **WebSocket** and
  speaks the **Cluster Control Protocol (CCP)** — register, heartbeat + telemetry, commands.
  See [02-control-plane](02-control-plane.md).
- The admin becomes a **scheduler + registry + monitor**: it knows every node's GPU/RDMA
  inventory live, schedules instances, coordinates mounts, and **detects and recovers dead
  nodes automatically**.
- Model storage becomes a **first-class role** served by **`modelfsd`**, a small userspace
  read-only NFS daemon, co-locatable on any node. See [03-storage-modelfsd](03-storage-modelfsd.md).

**Does not change — the data plane is proven, keep it:**

- Distributed execution still uses **Ray + NCCL over RoCE**. CCP *drives* Ray; it does not
  replace it. All the v0.3.0 NCCL/P2P/executor settings remain the defaults. See
  [04-data-plane](04-data-plane.md).
- vLLM itself is untouched — it still reads model files from a normal path and is launched with
  the same flags (including the hard-won `--enforce-eager` for cluster launches on Volta).
- The "nothing hardcoded" principle stands: fabric IPs, NIC names, HCA lists, GID indices stay
  in `.env`, now *reported by agents* rather than assumed.

## The architecture on one page

```
   ┌───────────────────────────── ADMIN (control plane) ─────────────────────────────┐
   │  FastAPI + UI        CCP WebSocket hub        Scheduler        Node registry      │
   │  (no GPUs required)  authenticated, inbound   TP/PP layout     live inventory     │
   └───────▲───────────────────▲──────────────────────────▲────────────────▲──────────┘
           │ CCP (WSS)          │ CCP (WSS)                │ CCP (WSS)       │
   heartbeat+telemetry   heartbeat+telemetry        heartbeat+telemetry     │
   commands ▼            commands ▼                 commands ▼              │
   ┌────────────────┐   ┌────────────────┐        ┌─────────────────────┐  │
   │  participant   │   │  participant   │        │  storage            │  │
   │  agent         │   │  agent         │        │  agent + modelfsd   │◄─┘ (admin may also
   │  Ray worker    │   │  Ray worker    │        │  read-only NFS/TCP  │     be storage)
   │  GPUs          │   │  GPUs          │        │  local model disk   │
   └───────┬────────┘   └───────┬────────┘        └──────────┬──────────┘
           │                    │                            │
           │  ═══ Ray control + NCCL tensor traffic over RoCE (DATA PLANE) ═══
           │                    │                            │
           └──── NFS mount (modelfsd, read-only) ── canonical model path ──┘

   Control plane  = CCP over WebSocket (orchestration, liveness, telemetry) — thin, typed
   Data plane     = Ray + NCCL over RoCE (tensors) + modelfsd NFS (weights) — fat, fast, proven
```

The guiding split: **control is thin and typed; data is fat and fast.** The WebSocket carries
kilobytes of JSON (who's alive, start this, mount that). The RoCE fabric carries the gigabytes
(activations between PP stages, weights off the storage node). They never mix.

## Why WebSocket for CCP (vs SSH, gRPC, a message bus)

| Option | Verdict |
|--------|---------|
| **SSH + remote shell** (PR #1) | Rejected. Keys to manage, inbound port on every worker, imperative, text-scraping, no liveness. |
| **WebSocket** | **Chosen.** Rides the existing FastAPI/uvicorn stack, ~zero new deps, **outbound-only from workers** (no inbound ports, firewall-friendly), one authenticated long-lived connection carries both telemetry and commands, trivial to add. |
| **gRPC** | Deferred. Real typed streaming, but adds protobuf toolchain + deps for a 2–4 node cluster; the message set is small enough that JSON-over-WS is enough. Revisit if the node count or message rate grows. |
| **Message bus (NATS/Redis)** | Rejected for now. Another moving part to run; overkill at this scale. |

## Glossary

- **Control plane** — the admin's orchestration brain: registry, scheduler, CCP hub, monitor.
- **Data plane** — everything that moves bulk bytes: Ray control, NCCL tensor traffic, `modelfsd` NFS.
- **CCP** — *Cluster Control Protocol*, the JSON-over-WebSocket protocol between agent and admin.
- **agent** — the per-node daemon that speaks CCP and executes commands locally.
- **role** — a capability a node offers: `admin`, `participant`, or `storage`. Composable.
- **instance** — one distributed vLLM instance (an OpenAI-compatible endpoint) spanning one or
  more participants with a given TP×PP layout.
- **`modelfsd`** — the read-only userspace NFSv3/TCP daemon serving model weights.
- **canonical model path** — the single filesystem path (e.g. `/export/llm_models`) at which
  every node sees the model repo; required because Ray/vLLM assume identical paths cluster-wide.
