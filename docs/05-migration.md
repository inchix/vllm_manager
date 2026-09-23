# 05 — Migration & compatibility

## Relationship to v0.3.0

v0.4.0 is **additive and opt-in**. v0.3.0's single-node and `CLUSTER_MODE` paths keep working
until the control plane is proven and the user chooses to switch.

- **Default off.** The control plane is gated by a flag (working name `CONTROL_PLANE=true`).
  With it unset, the manager behaves exactly as v0.3.0: local subprocess instances, and the
  existing `CLUSTER_MODE`/`ADMIN_ROLE` cluster path if configured.
- **Config maps forward, not backward:**
  | v0.3.0 | v0.4.0 |
  |--------|--------|
  | `ADMIN_ROLE=manager` | `ROLES=admin,participant[,storage]` |
  | `ADMIN_ROLE=worker` | `ROLES=participant[,storage]` + agent dials admin |
  | `ADMIN_ROLE=single` | `ROLES=admin,participant,storage` (one box) |
  | `CLUSTER_MODE=true` | `CONTROL_PLANE=true` |
  | manual `bash run.sh` per box | agents auto-register + reconcile |
  All fabric/NCCL vars (`NCCL_IB_HCA`, `NCCL_P2P_DISABLE`, `NCCL_SOCKET_IFNAME`, GID index, …)
  are **unchanged** and now reported by the agent at registration.
- **`setup.sh` gains role detection**: it already detects GPUs, NVLink, IOMMU, HCA, GID, NICs;
  it will additionally propose a `ROLES` set (a box with GPUs → `participant`; the box you run
  `setup.sh --role admin` on → `admin`; a box with the model disk → `storage`) and write the CCP
  vars. Nothing new is hardcoded.
- **Data plane unchanged** (see [04-data-plane](04-data-plane.md)), so an existing working
  cluster keeps the same NCCL/Ray/vLLM behaviour; only the orchestration around it changes.

## Relationship to community PR #1 (`cluster-two-host-ray`)

PR #1 and v0.4.0 want the **same thing** — a CPU-capable admin that need not hold GPUs, driving
GPU hosts — but differ on mechanism:

| Concern | PR #1 | v0.4.0 |
|---------|-------|--------|
| Admin→node control | **SSH** into Docker, run Ray, parse `nvidia-smi` text | **CCP** WebSocket, structured, outbound-only |
| Worker inbound ports | SSH open | none (agent dials out) |
| Liveness / recovery | none (manual) | heartbeat + reconcile (auto-rejoin) |
| GPU inventory | scrape `nvidia-smi` over SSH | typed `register`/`telemetry` |
| Model sync | rsync from controller with cluster key | `sync_model` command (+ optional rsync fan-out) |
| Storage | (kernel NFS assumed) | `modelfsd` role, co-locatable |
| Base | pre-v0.3.0 `main`; conflicts | branches from current `main` (== v0.3.0) |

**Plan:** treat PR #1 as a design input and a parts bin, not a merge. Cherry-pick the genuinely
good, reusable pieces onto the v0.4.0 model:

- ✅ **GPU selector shortcuts** (all / 50% / host-1 / host-2) → UI, fed by CCP inventory.
- ✅ **Gated-repo probe** (401/403 distinguished from a cache miss; token never logged) → folds
  into the `sync_model` command.
- ✅ **Preflight dependency check** (`deploy/check-deps.sh` / `/api/cluster/preflight`) → becomes
  an agent self-check reported at `register`.
- ✅ **Instance recovery after admin restart** → subsumed by the reconciliation loop.
- ⛔ **SSH-into-Docker orchestration** → replaced by CCP.
- ⛔ **`Containerfile.cluster` / `cluster/` Ray shell scripts** → replaced by the agent.

We should thank the contributor and, if they're willing, frame v0.4.0's admin/agent split as the
evolution of their idea. Do **not** attempt a straight merge onto v0.3.0.

## Upgrade path for the current lab (COVID + ebola)

1. Land Phase 1 (agent + CCP) behind `CONTROL_PLANE=true`.
2. COVID: `ROLES=admin,participant,storage`. ebola: `ROLES=participant`.
3. Bring COVID up (admin); ebola's agent dials in and registers.
4. Model repo served by `modelfsd` on COVID (or kept on kernel-RDMA per the benchmark) and
   mounted on ebola by `mount_storage`.
5. Launch an instance from the UI → scheduler fans out `ensure_instance` → same TP×PP×RoCE run as
   today, but now with auto-recovery if ebola blips.

Rollback is trivial: unset `CONTROL_PLANE` and you're back on the v0.3.0 path.

## Versioning

- Branch: **`v0.4.0`** (follows the repo's `v0.2.0`/`v0.3.0` milestone-branch convention).
- Kept in the `0.x` line deliberately: the cluster remains experimental hardware-specific work,
  not a 1.0 stability promise. The "major lift" is architectural, not a compatibility break —
  v0.3.0 behaviour is preserved behind the flag.
- Merge to `main` only after the Phase gates in [06-roadmap](06-roadmap.md) pass on real hardware.
