# 06 — Roadmap

Phased so each step is independently testable and nothing destabilises the working v0.3.0 path.
Every phase has an explicit **gate** — a check that must pass on the real COVID+ebola hardware
before moving on.

## Phase 0 — Storage transport benchmark (prerequisite, no code)

Decide the storage transport with a number, not a guess. See the benchmark plan in
[03-storage-modelfsd](03-storage-modelfsd.md).

- Measure current **kernel NFS-over-RDMA** cold-load of a real model (GB/s + vLLM cold
  `start_replica` wall-clock) from a participant.
- Stand up `modelfsd` (or a minimal NFSv3/TCP export) on the RoCE IP; measure the same.
- **Gate:** a decision recorded here — adopt `modelfsd` for `storage` if within ~1.5–2× of
  kernel-RDMA on cold start; otherwise keep kernel-RDMA transport under CCP coordination and
  defer `modelfsd`. Either way the rest of v0.4.0 proceeds (transport is pluggable).
- Runs on the current setup **without disturbing it** (ebola is back at full 300 W).

## Phase 1 — Control plane skeleton (agent + CCP + registry)

- Agent process (same image, `ROLES`/`CONTROL_PLANE` flags); dials admin WSS; `register` with
  GPU/RDMA inventory; `heartbeat` + `telemetry`.
- Admin: CCP hub endpoint (`/api/ccp`), node registry, `GET /api/cluster/nodes`, and a UI panel
  showing **GPUs grouped by host, live** (the long-promised cross-node monitor).
- No scheduling yet — observe only.
- **Gate:** COVID (admin) + ebola (participant) both visible with correct live GPU telemetry;
  kill ebola's agent / reboot ebola → admin marks it DOWN within `HEARTBEAT_MISS`, then READY
  again automatically on reconnect. **No inference involved** — pure control-plane liveness.

## Phase 2 — Scheduler + `ensure_replica` (single-node first)

- Admin scheduler computes layout (TP/PP/partition) and emits `ensure_replica`.
- Agent executes it: sets env, execs vLLM (reusing today's `vllm_manager` launch logic), reports
  health; reaps orphans and frees GPU reservations on stop/reconnect.
- Start with **single-node** (`node_count=1`) to validate the reconcile path end-to-end without
  the fabric.
- **Gate:** launch/stop a single-node replica on COVID purely via CCP; admin restart → replica
  re-adopted by the reconciler (no reload); orphaned-process/GPU-reservation leaks gone.

## Phase 3 — Storage role + mount coordination

- `serve_storage` (start `modelfsd` **or** manage the kernel-RDMA export per Phase 0) and
  `mount_storage`/`unmount_storage`; assert canonical path; gate `ensure_replica` on the mount.
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
- `setup.sh` role detection + CCP var generation.
- Docs: user-facing README section, update `CHANGELOG.md`, refresh `TODO.md`.
- **Gate:** a fresh two-box bring-up using only `setup.sh` + the UI, no manual `run.sh`.

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
