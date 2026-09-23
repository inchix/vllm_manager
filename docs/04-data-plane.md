# 04 — Execution / data plane

The data plane is what v0.3.0 already got working. v0.4.0 does **not** re-engineer it — it wraps
it. This doc records what stays fixed and where the control plane plugs in.

## What stays exactly as in v0.3.0

All of the hard-won hardware settings remain the defaults (in `run.sh` / `vllm_manager.py` /
`.env`), now *reported by agents* instead of assumed per box:

- **Distributed execution: Ray + NCCL.** Multi-node uses the Ray executor; single-node
  multi-GPU uses `mp`. Unchanged.
- **NCCL over RoCE**, with `NCCL_P2P_DISABLE=1` (broken PCIe P2P: no NVLink + IOMMU),
  `NCCL_IB_HCA=mlx4_0:1` (ConnectX-3 two-port cross-subnet fix), `NCCL_SOCKET_IFNAME` local-first
  comma list, single-NIC `GLOO_SOCKET_IFNAME`, and auto `--disable-custom-all-reduce`.
- **Volta constraints**: pinned CUDA-12/Volta-compatible base image; **`--enforce-eager` is
  mandatory for cluster launches** (CUDA-graph capture crashes `rc=1` on V100 + PP-over-RDMA —
  learned the hard way). The scheduler always includes it in `vllm_args` for a cluster replica.
- **Memory-weighted PP layer partition** across heterogeneous nodes (e.g. `27,13` for a
  32 GB : 16 GB split), driver-first ordering to match vLLM's stage assignment.
- **Confirmed RoCE RDMA path** for NCCL tensor traffic (verified via IB `port_xmit_data`
  counters); GPUDirect RDMA unsupported on ConnectX-3, so host-staged RDMA is the ceiling.

## Where the control plane plugs in

The admin's scheduler turns a user request ("serve model M across these nodes") into a set of
per-node `ensure_replica` commands. The agents do locally what `run.sh` + the manager did
before, but driven by CCP instead of by hand:

```
user picks model + nodes (UI)
        │
        ▼
admin scheduler:
  • read live inventory (GPUs per node, from telemetry)
  • compute layout: TP = gpus_per_node, PP = node_count
  • compute pp_layer_partition (memory-weighted)
  • pick head node, allocate port, assemble vllm_args (+ --enforce-eager)
        │
        ├── ensure_replica{role: head,   ray:{head_addr,port}, layout, vllm_args...} → node A agent
        └── ensure_replica{role: worker, ray:{head_addr,port}, ...}                  → node B agent
                                   │
      agents: ensure storage mounted ▼, start/join Ray, exec vLLM (head serves the API)
                                   │
              Ray + NCCL over RoCE ═══════════ tensor traffic ═══════════
```

The head node's vLLM process exposes the OpenAI-compatible endpoint on its allocated port,
exactly as today. The admin proxies/records it and reports health from `telemetry`.

## Ray placement-group lifecycle (the known leak)

Both our v0.3.0 experience **and** community PR #1 hit the same Ray failure mode:

> Stopping a replica without bouncing Ray can **leak the placement group**. A later start then
> fails with `ActorHandleNotFoundError` / `ActorDiedError` (an actor handle from Ray job *N*
> reused in job *N+1*). We also saw orphaned driver processes holding GPU memory after a remote
> node died.

v0.4.0 makes this manageable because the **control plane owns the Ray lifecycle**:

- `stop_replica` carries an explicit `ray: bool`. The admin's default teardown for a
  cluster replica is `ray: true` (bounce the Ray runtime with the replica) to avoid the leak;
  a lighter `ray: false` stop is available when reusing the runtime is known-safe.
- On node loss, the reconciler tears the replica down **and** clears Ray state on the survivors
  before any reschedule, so job *N+1* starts clean.
- The agent reaps orphaned vLLM/driver processes and frees GPU reservations on `stop_replica`
  and on reconnect-reconcile — closing the "orphaned process still holding GPUs / stuck
  `_used_gpus`" gap we hit manually in v0.3.0.

## Single-node is a degenerate cluster

A one-box `{admin,participant,storage}` node runs the same path with `node_count = 1`: PP=1,
TP=`gpus`, `mp` executor, `modelfsd` serving itself over the local path (no mount). No separate
"single vs cluster" code path in the launch logic — one scheduler, one reconciler, the layout
math collapses. This is a simplification over v0.3.0's `use_cluster` branch.

## Non-goals for the data plane in v0.4.0

- No change to NCCL/Ray/vLLM versions or the pinned base image (stability first).
- No GPUDirect RDMA (hardware can't).
- No attempt to replace Ray with a bespoke executor — CCP orchestrates Ray; rewriting the
  execution backend is explicitly out of scope.
