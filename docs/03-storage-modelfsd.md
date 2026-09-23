# 03 — Storage: `modelfsd`

`modelfsd` is the `storage` role's daemon: a **tiny, standalone, read-only NFSv3-over-TCP
server** that exports one local directory (the model repo) to the cluster. It replaces the
kernel `nfsd` + NFS-over-RDMA export that v0.3.0 uses today.

## Why not just keep kernel NFS-over-RDMA?

The v0.3.0 setup works, but it has three properties we want to be rid of:

1. **It needs the host kernel `nfsd` and privilege.** You cannot run it cleanly inside the
   hardened, non-root, read-only-rootfs admin container. Storage therefore can't be "just
   another role in the same image" — it's a host-level thing configured out of band
   (`/etc/exports`, `/etc/nfs.conf.d/rdma.conf`, `nfsd rdma 20049`). That fights the whole
   composable-roles model.
2. **A kernel NFS *client* wedges when the server disappears.** If the storage node hard-resets
   mid-read (the ebola failure), clients go into uninterruptible-sleep (D-state) on the mount.
   That can take out an otherwise-healthy participant.
3. **It's configured per-box by hand** — exactly the kind of hardcoded, manual setup v0.4.0's
   control plane is meant to eliminate.

## Why `modelfsd` fits

The reference is `wgshare/internal/nfsd` (a sibling project): a **hand-rolled, read-only NFSv3 +
MOUNTv3 server over ONC-RPC/TCP**, RFC-5531/1813 correct, **fuzz-tested**, no external NFS
dependency. Its properties map onto our problem almost exactly:

- **Userspace, non-root, single static (Go) binary.** Runs as an ordinary process in the same
  container image. Storage becomes a real composable role.
- **Read-only by design** — `WRITE` returns `NFS3ERR_ROFS`; it implements only what a read-only
  file server needs. Model weights are immutable, so this is a feature, not a limitation, and it
  shrinks the attack surface.
- **Prompt failure instead of D-state hang.** `wgshare/internal/lanserve` bounds each fetch with
  a `ReadTimeout` — a stalled source yields `NFS3ERR_IO` "rather than a client stuck in
  uninterruptible sleep." That is the ebola problem solved at the storage layer: a vanished
  storage node fails the in-flight replica but leaves the client box healthy and recoverable.
- **Path containment** (`wgshare/internal/safepath`) — the export root can't be escaped.

## What we take, and what we deliberately leave

`wgshare` solves a *different, bigger* problem (two households browsing each other's media over
WireGuard, fetching bytes from a remote peer on demand). Most of it is irrelevant to us. **We do
not copy it wholesale.**

**Port / adapt (the narrow wire core):**

- `nfsd` — the NFSv3/MOUNTv3/ONC-RPC/XDR server (read-only). This is the valuable, correctness-
  critical, fuzz-tested part — reuse beats rewrite.
- `safepath` — export-root containment.
- the **readahead** logic — the one performance feature that matters for bulk sequential reads.

**Drop entirely (the two-household machinery — we have none of it):**

- `index` / `catalogue` / `manifest` / `peersync` / `consume` — metadata indexing and
  remote-peer byte-fetch. **Our byte source is the local disk** (`os.Open` + `ReadAt`); there is
  no peer to fetch from and no metadata to sync.
- `invite` / `wg` / `tunnel` / `relay` / `fabric` / `roster` / `groups` — the WireGuard
  appliance and its pairing ceremony.
- `dlna` / `render` / `httpmedia` / `webui` / `pngmeta` — media browsing/presentation.

`lanserve`'s own comment says it *"joins the index and the peer transport to the NFS server."*
For us there is **no index and no peer transport**, so `lanserve` is replaced by a ~trivial
local-directory backend bolted directly under `nfsd`.

## Shape of the daemon

```
modelfsd --export /export/llm_models \
         --listen <fabric-ip>:2049 \
         --allow 172.16.254.0/24,172.16.253.0/24 \
         [--readahead 8m] [--metrics :9101]
```

- **One export, one local directory, read-only.** No config file, no state, no DB.
- **Backend = local filesystem.** A read is `ReadAt` on an `os.File` under the (contained)
  export root, fed through readahead. That's the whole data path.
- **Clients are the cluster's fabric IPs**, passed by the control plane (`--allow` / the
  `serve_storage` CCP command), not hand-maintained.
- Lives in this repo as a small Go module (e.g. `storage/modelfsd/`), built into the image
  alongside the Python admin — same single-binary shape wgshare already proves. The Python admin
  never speaks NFS; it just starts/stops/advertises the daemon over CCP.

## Client side — vLLM stays unchanged

Participants **kernel-mount** `modelfsd` at the canonical path (NFSv3 over TCP):

```
mount -t nfs -o vers=3,proto=tcp,ro,nofail,soft,timeo=100,retrans=3 \
      <storage-fabric-ip>:/export/llm_models  /export/llm_models
```

vLLM then reads `/export/llm_models/<model>/...` exactly as it does today — **no vLLM change**.
The storage node itself skips the mount and uses the directory locally. `soft`+`timeo` mean a
dead server returns errors rather than hanging the client (belt-and-braces with `modelfsd`'s own
`ReadTimeout`).

The control plane owns this: `mount_storage` issues the mount, asserts the canonical path
matches, and gates `start_replica` on the mount being present.

## The transport trade-off — and the benchmark that decides it

| | v0.3.0 today | `modelfsd` (v0.4.0) |
|---|---|---|
| Protocol | NFSv4.2 | NFSv3 |
| Transport | **RDMA** (kernel-bypass, zero-copy, port 20049) | **TCP** over the RoCE NIC's IP |
| Server | kernel `nfsd`, privileged, host-configured | userspace, non-root, in-container |
| Client hang on server loss | D-state | prompt `NFS3ERR_IO` |

Giving up RDMA transport costs CPU and zero-copy. **Whether that matters depends entirely on our
access pattern**, which is favorable:

- Model load is a **one-time bulk sequential read** at replica start (Devstral-24B ≈ 47 GB,
  streamed once), not the seek-heavy random IO where RDMA's latency win dominates.
- Even plain TCP over a 40–56 GbE-class RoCE NIC sustains multiple GB/s; readahead helps
  further. A ~47 GB load at, say, 2–3 GB/s is ~16–24 s, paid once per replica start.

**Adding a real RDMA transport to a userspace NFS server is exotic and a large lift — explicitly
out of scope for v1.** Instead we *measure* before committing:

### Benchmark plan (gates the decision, runs on the real hardware)

1. **Baseline** — current kernel NFS-over-RDMA: drop caches, time a full sequential read of a
   real model's weight shards from a participant (`dd`/`fio` sequential, plus a real vLLM cold
   `start_replica` wall-clock). Record GB/s and load seconds.
2. **Candidate** — `modelfsd` over RoCE-TCP: same model, same participant, same measurements.
3. **Compare on the metric that matters** — *cold replica start time* and sustained sequential
   GB/s. Decision rule: if `modelfsd` is within ~1.5–2× of kernel-RDMA on cold start, **adopt it**
   — the operational wins (containerable, non-root, no-hang, C2-driven, already fuzz-tested)
   dominate a one-time load cost. If it's dramatically worse, keep kernel-RDMA as the `storage`
   transport for now and revisit (the role/CCP design is transport-agnostic — see below).

This benchmark is Phase 0 / a prerequisite gate in [06-roadmap](06-roadmap.md), and can run on
the current setup **without disturbing anything** now that ebola is back at full power.

## Transport-agnostic escape hatch

The `storage` role and the `mount_storage` CCP command carry a `transport` field
(`rdma` | `tcp`). If the benchmark says keep RDMA for now, the control plane can drive **kernel
NFS-over-RDMA** as the storage transport while still owning mount coordination and liveness — we
get the orchestration wins immediately and adopt `modelfsd` when/if the numbers justify it.
`modelfsd` is thus an *upgrade of the storage backend*, not a prerequisite for the rest of v0.4.0.

## Security notes

- Read-only + `safepath` containment + an explicit client allow-list = small, auditable surface.
- No writes, no auth secrets, no state on the wire.
- Runs non-root; the export directory is mounted read-only into the container.
- Bind the listener to the **fabric** IP, not the management/public interface.
