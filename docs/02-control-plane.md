# 02 — Control plane (CCP)

The **Cluster Control Protocol** is the JSON-over-WebSocket protocol between each node's
**agent** and the **admin**. It is the whole of the control plane's wire format. This document
is the normative spec.

## Design principles

1. **Thin and typed.** CCP carries orchestration only — kilobytes of JSON. Never bulk data.
2. **Outbound from workers.** The agent *dials* the admin. Workers open **no inbound ports**;
   the only listening socket in the cluster is the admin's WSS endpoint on the management net.
3. **One connection, both directions.** Telemetry (agent→admin) and commands (admin→agent)
   share the single long-lived WebSocket. No second channel, no polling.
4. **Liveness is intrinsic.** The same connection that carries commands is the heartbeat. If it
   drops, the node is presumed down — no separate health check to get out of sync.
5. **Idempotent, declarative commands where possible.** `ensure_replica(spec)` rather than a
   fragile start/opaque-pid/stop dance, so a reconnecting agent can be driven back to the
   desired state without special-casing crash recovery.
6. **The agent owns the machine; the admin owns the intent.** The admin says *what* ("a replica
   with this layout"); the agent decides *how* locally (launch Ray worker, set env, exec vLLM)
   and reports back.

## Transport & framing

- **WebSocket over TLS** (`wss://`) to `https://<admin>:<ADMIN_PORT>/api/ccp` — same
  uvicorn/FastAPI server that already serves the admin UI and REST API.
- Each WebSocket **message** is one UTF-8 JSON object (a CCP *frame*). No binary frames in v1.
- Every frame has an envelope:
  ```json
  { "v": 1, "type": "<frame-type>", "id": "<uuid>", "ts": "<rfc3339>", "body": { ... } }
  ```
  - `v` — protocol version (integer). Mismatch → admin closes with a typed error (see below).
  - `type` — frame type (table below).
  - `id` — unique per frame; a reply echoes the request's `id` in `reply_to`.
  - `ts` — sender's clock (RFC 3339). Used for telemetry, **not** for auth (clocks are synced by
    chrony; nodes are set to a common timezone — see the ebola TZ note in memory).
- Commands that expect a result carry `id`; the agent answers with an `ack`/`result` frame
  whose `reply_to` = the command's `id`.

## Authentication & security

- The agent authenticates on connect with the **admin API key** (the existing
  `ADMIN_API_KEY`) via the `Authorization: Bearer` header on the WS upgrade request, plus a
  `cluster_id` and a stable `node_id`.
- TLS terminates at the admin. On a trusted RDMA/management fabric this may be a self-signed
  cert pinned by the agents; document both.
- The admin authorizes commands per role: an agent that only registered `participant` cannot be
  told to serve `storage`, etc. The key grants cluster membership; roles grant capabilities.
- **Secrets never traverse CCP in the clear and are never logged.** HF tokens for `sync_model`
  are sent over the TLS channel and redacted in all logs (carried over from PR #1's gated-repo
  handling).
- One `admin` per `cluster_id`. A second connection claiming `admin` for a live cluster is
  refused (`error: duplicate_admin`).

## Frame types

### Agent → Admin

| `type` | When | `body` (key fields) |
|--------|------|---------------------|
| `register` | First frame after connect | `node_id`, `hostname`, `roles[]`, `addresses{mgmt, fabric[]}`, `gpus[]` (index, model, mem_total, uuid), `rdma[]` (hca, ports, gid_index, netdev), `nccl_env{}` (the node's resolved NCCL/GID/HCA settings), `canonical_model_path`, `agent_version` |
| `heartbeat` | Every `HEARTBEAT_SEC` (default 5s) | `seq`, `uptime`, `load` |
| `telemetry` | Every `TELEMETRY_SEC` (default 5–10s) | `gpus[]` (index, util, mem_used, temp, power_draw, power_limit), `replicas[]` (id, state, port), `mounts[]` (path, source, ok) |
| `ack` | Immediately on receiving a command | `reply_to`, `accepted: bool`, `note?` |
| `result` | When a command finishes | `reply_to`, `ok: bool`, `state`, `detail?`, `error?` |
| `event` | Async local change (replica exited, mount dropped) | `kind`, `subject`, `detail` |

### Admin → Agent

| `type` | Purpose | `body` (key fields) |
|--------|---------|---------------------|
| `hello` | Accept registration | `assigned{}`, `heartbeat_sec`, `telemetry_sec`, `desired_state{}` (see reconciliation) |
| `ensure_replica` | Declare the desired replica shape this node participates in | `replica_id`, `role_in_replica` (`head`/`worker`), `ray{head_addr, port}`, `model`, `layout{tp, pp, pp_layer_partition?}`, `vllm_args[]` (incl. `--enforce-eager`), `port`, `env{}` |
| `stop_replica` | Tear down a replica on this node | `replica_id`, `ray: bool` (also stop the Ray runtime — see PG-leak note) |
| `mount_storage` | Ensure the model repo is mounted | `source{host, export, transport}`, `canonical_path`, `opts[]` |
| `unmount_storage` | Drop a mount | `canonical_path` |
| `serve_storage` | (storage role) start/ensure `modelfsd` | `export_dir`, `allow[]` (client fabric IPs), `listen{fabric_ip, port}` |
| `sync_model` | Fetch a model to local disk (storage/participant) | `repo`, `revision`, `dest`, `hf_token?` (redacted), `then_rsync_to[]?` |
| `error` | Protocol/authorization failure | `code`, `detail` |
| `bye` | Graceful admin shutdown / eviction | `reason` |

## Reconciliation loop (why commands are declarative)

The admin holds a **desired state** per node and per replica. On every (re)connect it sends the
node's `desired_state`, and the agent reconciles: start what should be running, stop what
shouldn't. This is what makes recovery boring:

```
ebola resets ──► WS drops ──► admin marks ebola DOWN, marks its replicas FAILED,
                              (optionally reschedules onto remaining nodes)
ebola reboots ─► agent reconnects, re-registers ──► admin resends desired_state
                              ──► agent re-mounts storage, rejoins Ray, restarts its
                                  replica half ──► replica HEALTHY again, no human
```

Contrast v0.3.0, where this whole path was a person typing `bash run.sh` and re-launching the
model from the UI. **The reconciliation loop is the headline feature.**

## Node lifecycle (admin's view)

```
                register ok
   (connect) ─────────────────► READY ──ensure_replica──► SERVING
      ▲                          │  ▲                        │
      │                    miss  │  │ reconnect+reconcile    │ replica exits / heartbeat gap
      │ reconnect          N HB  ▼  │                        ▼
   DISCONNECTED ◄──────────── DOWN ◄─────────────────────── DEGRADED
```

- **READY** — registered, idle, inventory known.
- **SERVING** — running its half of ≥1 replica.
- **DEGRADED** — a replica on this node failed but the node is alive (e.g. vLLM crashed but the
  box is fine); admin retries per policy.
- **DOWN** — `HEARTBEAT_MISS` intervals (default 3) with no heartbeat → presumed reset/dead;
  replicas depending on it are marked FAILED.
- **DISCONNECTED → reconnect** re-enters the reconcile path.

## Telemetry = free cluster-wide monitoring

The `telemetry` frame is exactly the per-node GPU stream we sketched earlier (util/mem/temp/
power per GPU, plus replica and mount state). The admin aggregates it into:

- `GET /api/cluster/nodes` — every node, its roles, state, GPUs, replicas (JSON for the UI).
- A UI panel grouping GPUs **by host**, live, nothing hardcoded — the "monitor GPU usage across
  all nodes" ask, delivered as a side effect of the protocol rather than a bespoke feature.

## Tunables (all in `.env`, defaulted)

| Var | Default | Meaning |
|-----|---------|---------|
| `CCP_HEARTBEAT_SEC` | 5 | Agent heartbeat interval |
| `CCP_TELEMETRY_SEC` | 10 | Telemetry interval |
| `CCP_HEARTBEAT_MISS` | 3 | Missed heartbeats before a node is DOWN |
| `CCP_RECONNECT_BACKOFF` | 1..30s | Agent reconnect backoff (jittered) |
| `CLUSTER_ID` | `default` | Cluster identity the agent joins |

## Open questions (resolve during Phase 1)

- **Rescheduling policy** on node loss: auto-move a replica to survivors, or hold and alert?
  Start with *hold + alert* (deterministic), add auto-reschedule later.
- **Multiple replicas per node** (port allocation, GPU partitioning) — the frame set supports it;
  the scheduler's packing logic is a Phase 2 concern.
- **mTLS vs bearer key** on the fabric — bearer key for v1, leave a hook for client certs.
