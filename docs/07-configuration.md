# 07 — Configuration & the per-node override model

Two hard requirements drive this document:

1. **Everything is editable in the UI.** Every knob that today lives only in `.env` must be
   settable from the admin UI — not just model/GPU/util, but the fabric and hardware-workaround
   settings too (NCCL, executor, P2P, GID, power caps, roles, ports).
2. **Config is per-node overridable.** Worker nodes are *not* identical. The cluster must carry a
   base config plus **per-node overrides**, because real boxes differ — exactly as COVID and
   ebola already did in v0.3.0.

## Why per-node is mandatory (this is not hypothetical)

The v0.3.0 lab already needed different settings per box. From the two `.env` files:

| Setting | COVID | ebola | Why it differs |
|---------|-------|-------|----------------|
| `NCCL_SOCKET_IFNAME` | `enp196s0,ens2` | `ens2,enp196s0` | NIC names differ; **local NIC must be first** on each box |
| `GLOO_SOCKET_IFNAME` | `enp196s0` | `ens2` | single local NIC, per box |
| `NCCL_P2P_DISABLE` | `1` | (varies) | COVID has IOMMU-translated P2P broken; ebola's IOMMU state differs |
| `RAY_NODE_IP` | COVID mgmt IP | ebola mgmt IP | per box |
| GPU power cap | 300 W | 150 W (workaround era) | per box, per hardware condition |
| `ADMIN_ROLE` | manager | worker | per box |

A single flat cluster config **cannot** express this. So per-node override isn't a nice-to-have;
it's required for the cluster to run at all on heterogeneous hardware.

## The layered config model

Effective config for a node is a **merge of four layers**, lowest to highest precedence:

```
  4. per-node override        (admin UI, per worker)          ── highest precedence
  3. cluster defaults         (admin UI, applies to all nodes)
  2. node-detected values     (agent auto-detects: NICs, GID, HCA, GPUs, IOMMU, NVLink)
  1. image defaults           (safe neutral defaults baked into the image)  ── lowest
        │
        ▼
   effective config for THIS node  =  merge(1,2,3,4)
```

- **(1) Image defaults** — the neutral, "nothing hardcoded" defaults already in `run.sh`/`.env.example`.
- **(2) Node-detected** — the agent runs the *same detection logic as `setup.sh`* (GPU count,
  NVLink, IOMMU, HCA + ports, RoCEv2 GID index, RoCE NIC, mgmt IP) and reports it at `register`.
  This means a fresh worker often needs **zero** manual config — it detects its own fabric.
- **(3) Cluster defaults** — set once in the UI, applied to every node (e.g. default GPU util,
  model repo path, heartbeat intervals, default vLLM args).
- **(4) Per-node override** — set in the UI for a specific worker, wins over everything. This is
  where ebola-specific NIC ordering or a power cap lives.

**Detected vs pushed.** Some values flow *up* (agent detects → admin), some flow *down* (admin →
agent). The override layer lets the admin **correct or pin** a detected value (e.g. force a NIC
order detection got wrong). Resolution is always: detected is the starting point, override wins
if present.

### Where it's stored and how it moves

- The **admin holds the authoritative cluster config** (cluster defaults + per-node overrides),
  persisted alongside the existing instance persistence.
- On registration and on any change, the admin computes each node's **effective config** and
  sends it in the CCP `hello` / `desired_state` (see [02-control-plane](02-control-plane.md)).
- The agent applies it locally (exports the env for Ray/NCCL/vLLM, sets power caps, etc.) when it
  starts an instance.
- `.env` still works as the **bootstrap** for a node (how the agent finds the admin and its
  `cluster_id`/roles before it has ever connected) and as the fallback when the control plane is
  off. Once connected, admin config takes precedence for the managed settings.

## Settings catalog — cluster-wide vs per-node

Every one of these is exposed in the UI. "Scope" says where it naturally lives; per-node ones are
the override surface.

### Identity & roles (per-node)
- `ROLES` (`admin`/`participant`/`storage` set), `CLUSTER_ID`, `RAY_NODE_IP`, fabric IP(s),
  `RAY_HEAD_HOST`/port (workers).

### Fabric / NCCL (per-node — the heterogeneity that matters)
- `NCCL_SOCKET_IFNAME` (comma list, local-first), `GLOO_SOCKET_IFNAME` (single local NIC),
  `NCCL_IB_HCA` (e.g. `mlx4_0:1`), `NCCL_IB_GID_INDEX`, `NCCL_IB_DISABLE`, `NCCL_P2P_DISABLE`,
  RDMA device passthrough list.
- Each field shows its **detected value** with the override empty by default, so you only fill in
  what's actually different.

### Execution / vLLM (mostly cluster-wide, some per-node)
- Cluster-wide: default `gpu_memory_utilization`, `MULTIGPU_EXECUTOR` (mp/ray),
  `DISABLE_CUSTOM_ALL_REDUCE`, default `--enforce-eager` for cluster launches, default
  `dtype`/`max_model_len`, tool-use defaults, port range, compile-cache dir.
- Per-node (rare but supported): executor override, extra vLLM args for a specific box.

### Storage (per-node role config)
- `storage`: export dir, listen fabric IP + port, client allow-list, readahead.
- clients: canonical mount path, mount opts, transport (`tcp`/`rdma`).

### Hardware guards (per-node)
- GPU power cap (W) + persistence mode — *promoted from a manual `nvidia-smi -pl` to a first-class
  per-node setting the agent applies at start* (the ebola workaround becomes a config field).
- Optional per-GPU enable/disable.

### Control plane (cluster-wide)
- `CCP_HEARTBEAT_SEC`, `CCP_TELEMETRY_SEC`, `CCP_HEARTBEAT_MISS`, `CCP_RECONNECT_BACKOFF`.

## UI shape

```
Cluster ▸ Configuration
┌───────────────────────────────────────────────────────────────────────┐
│ [ Cluster defaults ]   applies to all nodes                            │
│   util 0.85 · executor mp · enforce-eager ✔ · model path /export/... │
│                                                                         │
│ [ Nodes ]   tabs: (COVID · admin,participant,storage) (ebola · part.)  │
│  ┌── ebola ───────────────────────────────────────────────────────┐   │
│  │ setting              detected            override                │   │
│  │ NCCL_SOCKET_IFNAME   ens2,enp196s0       [ens2,enp196s0    ]     │   │
│  │ GLOO_SOCKET_IFNAME   ens2                [                 ]     │   │
│  │ NCCL_P2P_DISABLE     (auto)              [1 ▾]                    │   │
│  │ GPU power cap (W)    300                 [        ] persist ☐    │   │
│  │ roles                                    [participant ▾]         │   │
│  │ … every setting, grouped: Identity · Fabric · Execution · HW …  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│  Effective config (merged, read-only preview) · [Save]                  │
└───────────────────────────────────────────────────────────────────────┘
```

- **Per-node tabs/cards**, one per registered node, each with the full settings list.
- Each field shows **detected value** (from the agent) next to an **override** input — empty
  override = "use detected/default." This keeps the common case (fresh worker) empty and makes the
  *differences* the only thing you type.
- An **effective-config preview** shows the merged result before saving, so there's no guessing.
- Save pushes to the admin; the admin recomputes effective config and sends it to the affected
  agents. Changes take effect on the **next instance start** for launch-time settings; live-safe
  settings (e.g. power cap) can apply immediately.
- `setup.sh` still seeds a sensible starting config; the UI is where you refine per-node.

## Validation & safety

- The admin validates overrides (e.g. `GLOO_SOCKET_IFNAME` must be a single interface; a NIC must
  exist in the node's detected list, or be flagged as a manual pin).
- Secrets (HF token) are write-only fields, never echoed back, never logged (carried from PR #1).
- An invalid per-node override is rejected at save with the reason, not discovered at instance
  launch.
- Read-only "effective config" export per node (for support/repro), with secrets redacted.
