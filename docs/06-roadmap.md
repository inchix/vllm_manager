# 06 — Roadmap

Phased so each step is independently testable and nothing destabilises the working v0.3.0 path.
Every phase has an explicit **gate** — a check that must pass on the real COVID+ebola hardware
before moving on.

## Phase 0 — Storage transport decision (settled, no load test)

**Decided: `modelfsd` over RoCE-TCP for the `storage` role.** No benchmark gate — model load is a
one-time bulk sequential read where TCP-over-RoCE is comfortably fast, and the operational wins
dominate (see [03-storage-modelfsd](03-storage-modelfsd.md)). The transport stays **pluggable**
(kernel-RDMA remains a fallback the control plane can drive), so if a *real* production
bottleneck ever appears we revisit with actual numbers rather than a synthetic test.

## Phase 1 — Control plane skeleton (agent + CCP + registry) — ✅ IMPLEMENTED

- ✅ Agent process (`admin/agent/agentd.py`, same image, `ROLES`/`CONTROL_PLANE` flags); dials
  the admin over the CCP WebSocket; `register` with detected GPU/RDMA inventory; `heartbeat` +
  live `telemetry`. Reconnects with jittered backoff and re-registers.
- ✅ Admin: CCP hub (`admin/cluster/hub.py`, `/api/ccp`), node registry with lifecycle states,
  `GET /api/cluster/nodes` / `/summary` / `/config`, mounted in `app.py` behind `CONTROL_PLANE`.
- ✅ Layered config with per-node overrides (`admin/cluster/config.py`) and hardware detection
  (`admin/cluster/detect.py`). UI panels (`admin/static/cluster.*`) render live GPUs-by-host and
  the per-node config editor.
- ✅ **Verified** by `admin/cluster/smoke_test.py`: a real agent registers against a real hub and
  streams live per-GPU telemetry (util/mem/temp/power) — 7/7 checks pass on the COVID hardware.
- **Remaining gate (needs both boxes):** COVID (admin) + ebola (participant) both visible with
  live telemetry; reboot ebola → admin marks it DOWN within `HEARTBEAT_MISS`, then READY again
  automatically on reconnect. (The transition logic exists; the two-box run is the field check.)

## Phase 2 — Scheduler + `ensure_replica` (single-node first) — 🟡 logic done, HW gate pending

- Admin scheduler computes layout (TP/PP/partition) and emits `ensure_replica`.
- Agent executes it: sets env, execs vLLM (reusing today's `vllm_manager` launch logic), reports
  health; reaps orphans and frees GPU reservations on stop/reconnect.
- Start with **single-node** (`node_count=1`) to validate the reconcile path end-to-end without
  the fabric.
- ✅ Scheduler (`scheduler.py`) computes layout + memory-weighted partition + executor; runner
  executes it (mp single-node / Ray multi-node); `POST /api/cluster/plan` dry-run; the v0.3.0
  executor lessons encoded (mp single-node, Ray+`--enforce-eager` only multi-node,
  `--disable-custom-all-reduce` for any multi-GPU). Unit-tested (`test_scheduler.py`, 10).
- **Gate (needs container + GPUs):** launch/stop a single-node replica on COVID purely via CCP;
  admin restart → replica re-adopted by the reconciler; orphaned-process/GPU-reservation leaks
  gone.

## Phase 3 — Storage role + mount coordination — 🟡 logic done, HW gate pending

- ✅ `serve_storage` (starts `modelfsd`) + `mount_storage`/`unmount_storage` in the runner;
  `hub._coordinate_storage()` serves on the storage node and mounts on each participant at the
  canonical path before a replica loads, gating the launch on it.
- Storage co-located on COVID; ebola mounts over the fabric.
- **Gate:** ebola loads a model entirely through the coordinated mount; storage-node-down is
  detected and blocks new replicas with a clear message; a client read against a killed server
  errors promptly (no D-state) — verified by killing the server mid-read.

## Phase 4 — Multi-node replica + auto-recovery (the payoff)

- Fan-out `ensure_replica` across COVID+ebola: TP=2 × PP=2, `--enforce-eager`, memory-weighted
  partition, Ray+NCCL over RoCE — the v0.3.0 run, now control-plane-driven.
- Reconciler handles node loss: mark FAILED, clear Ray PG on survivors, auto-rejoin on
  reconnect.
- **Gate:** serve Devstral at 64k across both boxes via CCP; reboot ebola under load → replica
  fails, ebola auto-rejoins on boot and the reconciler restores the replica **with no human
  action**. This is the headline demo and the definition of done for the core lift.

## Phase 5 — Polish & PR #1 cherry-picks

- GPU selector shortcuts, gated-repo probe, preflight-as-agent-selfcheck (from
  [05-migration](05-migration.md)).
- **Full cluster-config UI with per-node overrides** — every setting editable in the UI (not
  just `.env`), cluster-wide defaults plus per-worker overrides, effective config pushed to each
  agent over CCP. See [07-configuration](07-configuration.md). (The config *model* — layered
  resolution, detected-vs-pushed settings — is used from Phase 1 onward; this phase completes the
  editing surface.)
- `setup.sh` role detection + CCP var generation.
- Docs: user-facing README section, update `CHANGELOG.md`, refresh `TODO.md`.
- **Gate:** a fresh two-box bring-up using only `setup.sh` + the UI, no manual `run.sh`; and
  editing a per-worker setting (e.g. ebola's `NCCL_SOCKET_IFNAME`) in the UI takes effect on the
  next replica start without touching any file.

## Phase 6 — Merge

- Rebase/settle onto `main`; ensure `CONTROL_PLANE` unset == exact v0.3.0 behaviour.
- **Gate:** v0.3.0 regression pass (single-node + legacy cluster) green with the flag off; then
  fast-forward/merge `v0.4.0` → `main`.

## Definition of done (the whole lift)

> From the UI, launch a model across COVID+ebola with no SSH and no manual per-box commands;
> watch live per-host GPU usage; reboot ebola mid-inference; watch the cluster mark it down and
> **auto-heal when it returns** — all driven by the control plane, with the proven RoCE data
> plane underneath.

## Explicit non-goals for v0.4.0

- RDMA transport *inside* `modelfsd` (userspace verbs NFS) — out of scope; kernel-RDMA remains
  the fallback transport.
- Replacing Ray/NCCL/vLLM or the pinned base image.
- Auto-rescheduling replicas onto survivors (start with hold+alert; revisit later).
- Multi-storage failover, GPUDirect RDMA, >2-box packing heuristics — designed-for, not built.
