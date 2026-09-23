# vLLM Manager — v0.4.0 design docs

This directory is the design record for the **v0.4.0 re-architecture**: a re-think of how
vLLM Manager spans multiple machines. v0.3.0 proved that distributed inference *works* on this
hardware (multi-node TP×PP over RoCE RDMA, shared models over NFS/RDMA). v0.4.0 keeps that
proven **data plane** and puts a real **control plane** on top of it, replacing ad-hoc
SSH/manual orchestration with a small command-and-control protocol and composable node roles.

> Status: **design / not yet implemented.** These documents are the spec we build against.
> Nothing here changes the running v0.3.0 behaviour until the phases in
> [06-roadmap](06-roadmap.md) land behind a feature flag.

## Read in order

| Doc | What it covers |
|-----|----------------|
| [00 — Overview](00-overview.md) | Why v0.4.0, what changes from v0.3.0, the whole architecture on one page, glossary |
| [01 — Composable roles](01-roles.md) | `admin` / `participant` / `storage` as a freely-composed set per node |
| [02 — Control plane (CCP)](02-control-plane.md) | The WebSocket command-and-control protocol: registration, heartbeat/telemetry, commands, lifecycle, failure handling, security |
| [03 — Storage: `modelfsd`](03-storage-modelfsd.md) | The embedded read-only NFS daemon for model weights; transport trade-off and benchmark plan |
| [04 — Execution / data plane](04-data-plane.md) | Ray + NCCL over RoCE, unchanged, now *driven by* the control plane; placement-group lifecycle |
| [05 — Migration & compatibility](05-migration.md) | Relationship to v0.3.0 `CLUSTER_MODE` and to community PR #1; upgrade path; config |
| [06 — Roadmap](06-roadmap.md) | Phased implementation plan with test/benchmark gates |

## One-paragraph summary

A node advertises a **set of roles** — any of `admin`, `participant`, `storage` — and runs one
lightweight **agent**. Every agent dials the **admin control plane** over an authenticated,
outbound **WebSocket** (the *Cluster Control Protocol*, "CCP"), registers its GPU/RDMA
inventory, streams heartbeat + telemetry, and receives typed commands (`start_replica`,
`join_ray`, `mount_storage`, `sync_model`, `stop`). The admin is the brain; it schedules
distributed vLLM replicas onto participants, coordinates model mounts from `storage` nodes, and
detects a dead node the instant its heartbeat stops — so a box that hard-resets (see the ebola
saga) **auto-rejoins on reconnect** instead of needing a manual restart. The actual tensor
traffic still rides **Ray + NCCL over RoCE**, and model bytes are served by **`modelfsd`**, a
tiny userspace **read-only NFSv3/TCP** daemon (protocol core ported from the fuzz-tested
`wgshare/internal/nfsd`) that any node can run — solving both the "can't run `nfsd` in a
container" problem and the "kernel NFS client wedges in D-state when the server vanishes"
problem in one move.
