# 01 — Composable roles

## The model

In v0.3.0 a node has exactly one `ADMIN_ROLE`: `manager`, `worker`, or `single`. That is too
rigid — it can't express "the admin box also has GPUs and also holds the models," which is
literally how the single-box COVID setup already runs.

v0.4.0 replaces the single role with a **set** of roles a node advertises:

```
roles = { admin?, participant?, storage? }        # any non-empty subset
```

The three roles are orthogonal capabilities:

| Role | Provides | Needs |
|------|----------|-------|
| **admin** | The control plane: FastAPI + UI, the CCP WebSocket hub, the scheduler, the node registry. Exactly **one** node per cluster is admin. | CPU + a reachable address on the management network. **No GPUs required.** |
| **participant** | GPUs for inference. Runs the agent and, when scheduled, a Ray worker + the local half of a vLLM replica. **N** per cluster. | GPUs, RDMA NIC on the fabric, the model visible at the canonical path. |
| **storage** | The model repo, exported read-only over the fabric via `modelfsd`. **≥1** per cluster (see failure coupling below). | Local disk holding the models, an RDMA NIC. |

A node's roles are set in its `.env` (or by `setup.sh` detection) and reported to the admin at
registration. Nothing about the layout is hardcoded.

## Example topologies

All four are the *same software*, just different role compositions:

```
Single box (COVID today)          admin + participant + storage on one node
┌─────────────────────────────┐
│ COVID: {admin,participant,   │   the whole cluster is one machine;
│         storage}             │   modelfsd serves the local disk to itself (local path)
└─────────────────────────────┘

Two GPU boxes, storage on one     the v0.3.0 shape, done right
┌──────────────────┐  ┌───────────────────────────┐
│ COVID:           │  │ ebola:                     │
│ {admin,          │  │ {participant}              │
│  participant,    │◄─┤  mounts models from COVID  │
│  storage}        │  │                            │
└──────────────────┘  └───────────────────────────┘

CPU admin + 2 GPU boxes           PR #1's goal, without SSH
┌───────────┐  ┌──────────────┐  ┌──────────────┐
│ ctl:      │  │ gpu-a:       │  │ gpu-b:       │
│ {admin,   │  │ {participant}│  │ {participant,│
│  storage} │  │              │  │  storage}    │  ← storage co-located on a worker
└───────────┘  └──────────────┘  └──────────────┘

Dedicated everything
┌────────┐ ┌──────────────┐ ┌──────────────┐ ┌───────────┐
│ {admin}│ │ {participant}│ │ {participant}│ │ {storage} │
└────────┘ └──────────────┘ └──────────────┘ └───────────┘
```

## Storage is co-locatable anywhere

This is the key flexibility. `storage` has **no coupling** to `admin` or `participant`; it's
just another flag. It can sit:

- on the **admin** box (natural when the admin has the big disk — today's COVID),
- on a **participant** box (efficient: it already has the RDMA NIC and reads its own models
  locally with no NFS hop),
- on a **dedicated** box.

The storage host reads the model repo **locally** (direct path); every other node **mounts it
over the fabric** via `modelfsd`. Everyone ends up with the repo at the same *canonical path*,
which Ray/vLLM require.

## Failure coupling — the one placement rule

If `storage` lives on a `participant` and **that box resets**, its export vanishes and every
peer loses the mount mid-inference. Given the hardware history (ebola hard-reset repeatedly
before its PSU was fixed), the guidance is blunt:

> **Put `storage` on your most reliable node.** Co-locate it on a participant only if that box
> is proven stable. Never put the sole `storage` role on your flakiest box.

The control plane helps but cannot repeal physics:

- The CCP heartbeat detects storage-node loss within one interval, so the admin can **refuse to
  schedule replicas that depend on a down export** and surface the outage immediately.
- Client mounts use `nofail` (already set in the v0.3.0 fstab guidance) so a peer never hangs
  at boot on a missing export.
- `modelfsd`'s userspace design means a client read against a *gone* server returns a prompt
  error rather than a D-state hang (see [03-storage-modelfsd](03-storage-modelfsd.md)) — the
  in-flight replica still fails, but the box stays healthy and recovers cleanly.

Future option (not v1): **multiple storage nodes** with the same repo (replicated or
read-through), so the admin can fail mounts over. Designed for, not built yet.

## Constraints & invariants

- **Exactly one `admin`** per cluster. Two admins is a split-brain; the CCP handshake rejects a
  second admin claiming the same cluster id.
- **At least one `storage`** reachable, or no replica can load weights. A cluster with zero
  storage is legal only for nodes that already have models on local disk at the canonical path.
- **The canonical model path is identical on every node** — enforced by the admin when it
  issues `mount_storage`, and asserted before `start_replica`.
- A node with **no `participant` role contributes no GPUs**, even if it physically has them
  (lets you keep the admin box's GPUs free for other work — e.g. the Ollama coexistence we
  needed on COVID).
