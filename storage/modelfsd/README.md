# modelfsd

A tiny, standalone, **read-only NFSv3-over-TCP** file server. It exports **one local
directory** (the model repo) to the cluster fabric and nothing else — no writes, no config
file, no state, no database. It is the `storage` role's daemon described in
[`docs/03-storage-modelfsd.md`](../../docs/03-storage-modelfsd.md).

It runs as an ordinary non-root process: a single static Go binary, servable from inside
the hardened admin container image, replacing the kernel `nfsd` + NFS-over-RDMA export that
v0.3.0 configured out of band.

## CLI

```
modelfsd --export <dir> --listen <ip:port> --allow <cidr,cidr,...> \
         [--readahead 8m] [--metrics <ip:port>] [--export-name <path>]
```

| Flag | Required | Meaning |
|------|----------|---------|
| `--export` | yes | Absolute path of the directory to serve, read-only. |
| `--listen` | yes | Address to bind, `ip:port` — bind the **fabric** IP, e.g. `172.16.254.10:2049`. |
| `--allow` | yes | Comma-separated client CIDRs (or bare IPs) permitted to mount and read. A client outside every network is refused at MNT and on every op. |
| `--readahead` | no (default `8m`) | Read-ahead window per stream. Accepts `k`/`m`/`g` suffixes (powers of 1024); `0` disables prefetch. |
| `--metrics` | no | If set, expose Prometheus-style counters over HTTP at this `ip:port` (`/metrics`). |
| `--export-name` | no | The mount path clients spell. Defaults to the `--export` path, so `--export /export/llm_models` is mounted as `<ip>:/export/llm_models`. |

Example:

```
modelfsd --export /export/llm_models \
         --listen 172.16.254.10:2049 \
         --allow 172.16.254.0/24,172.16.253.0/24 \
         --readahead 8m --metrics 127.0.0.1:9101
```

### Client side (unchanged from v0.3.0)

Participants kernel-mount it, NFSv3 over TCP:

```
mount -t nfs -o vers=3,proto=tcp,ro,nofail,soft,timeo=100,retrans=3 \
      172.16.254.10:/export/llm_models  /export/llm_models
```

`soft`+`timeo` make a dead server return errors rather than hanging the client —
belt-and-braces with modelfsd's own read timeout.

## NFS version / transport

- **NFSv3** (RFC 1813) + **MOUNTv3** (served in-process on the same port — no rpcbind, no
  `rpc.mountd`, no `rpc.statd`), over **ONC-RPC v2 / TCP** (RFC 5531).
- **TCP only.** No UDP, no RDMA transport (see the transport trade-off in the design doc).
- **AUTH_NULL / AUTH_SYS** are accepted; both are squashed to nobody. Authorisation is
  decided **only** by the client's address against `--allow`, never by a claimed uid.
- **Read-only:** `WRITE`, `SETATTR`, `CREATE`, `MKDIR`, `SYMLINK`, `MKNOD`, `REMOVE`,
  `RMDIR`, `RENAME`, `LINK`, `COMMIT` all return `NFS3ERR_ROFS` with the correct failure
  body for their result type. The implemented reads are `GETATTR`, `LOOKUP`, `ACCESS`,
  `READLINK`, `READ`, `READDIR`, `READDIRPLUS`, `FSSTAT`, `FSINFO`, `PATHCONF`, `NULL`.

## What was ported vs dropped

Ported and adapted from the sibling project **wgshare**
(`/home/adwhite/dev/wireguard_router`, same owner):

| Ported | From | Notes |
|--------|------|-------|
| `internal/nfsd` | `wgshare/internal/nfsd` | The NFSv3 / MOUNTv3 / ONC-RPC v2 / XDR read-only server. RFC-correct, bounds-checked, fuzz-tested. Taken almost verbatim; only the package doc and `shareFromPath` changed (see caveats). |
| `internal/safepath` | `wgshare/internal/safepath` | Export-root containment (openat2 `RESOLVE_BENEATH`, with a component-walk fallback). Verbatim apart from the resolver-override env var name. |
| `internal/localfs` (read-ahead) | `wgshare/internal/lanserve/readahead.go` | The chunked read-ahead + singleflight + read-timeout logic, re-pointed from a remote peer transport to a local `*os.File`. |

Replaced:

- **`lanserve`'s index + peer byte-source** → `internal/localfs`, a trivial local-directory
  backend. A read is `ReadAt` on an `os.File` under the (safepath-contained) export root,
  fed through read-ahead. There is **no remote peer and no metadata index** — the files are
  on local disk.

Dropped entirely (none of it is relevant here): `index`, `catalogue`, `manifest`,
`peersync`, `consume`, `invite`, `wg`, `tunnel`, `relay`, `fabric`, `roster`, `groups`,
`dlna`, `render`, `httpmedia`, `webui`, `pngmeta` — the two-household / WireGuard / media
machinery.

## The no-hang property

Every physical read is bounded by a read timeout (default 20s). A stalled or vanished disk
yields `NFS3ERR_IO` promptly instead of a client wedged in uninterruptible sleep — the
ebola failure solved at the storage layer (design doc §"Why modelfsd fits").

## Metrics

With `--metrics <ip:port>`, `GET /metrics` returns Prometheus text: `modelfsd_rpc_ops_total`
and `modelfsd_rpc_ops_by_proc_total{proc=…}`, `modelfsd_rpc_errors_total` (RPC-level
failures), `modelfsd_read_bytes_total`, and `modelfsd_uptime_seconds`.

## Layout

```
storage/modelfsd/
├── go.mod                         module github.com/inchix/vllm_manager/storage/modelfsd
├── README.md
├── cmd/modelfsd/main.go           CLI: flags, listener, signal handling, metrics endpoint
└── internal/
    ├── nfsd/                       ported: NFSv3/MOUNTv3/ONC-RPC/XDR read-only server
    │   ├── nfsd.go  server.go  rpc.go  xdr.go  nfs.go  mount.go  observe.go
    ├── safepath/                   ported: export-root containment
    │   ├── api.go  root.go  validate.go  resolve_openat2.go  resolve_walk.go
    ├── localfs/                    new: local-directory backend + read-ahead
    │   ├── backend.go  readahead.go
    │   └── *_test.go               backend, read-path, escape and over-the-wire tests
    └── metrics/metrics.go          new: counters + HTTP exposition
```

## Build / test

```
cd storage/modelfsd
go build ./...
go vet ./...
go test ./...
```

Requires Go 1.24. `golang.org/x/sys` is pinned to `v0.37.0` (later releases require Go ≥
1.26); it is the only external dependency, used by `safepath` for `openat2`/`fstatat`.

## Porting caveats / intentionally omitted

- **`shareFromPath` was relaxed.** wgshare exported a single name at the root (`/films`);
  modelfsd's one export has a multi-segment canonical mount path (`/export/llm_models`), so
  the MOUNT dirpath is normalised and kept whole (interior `/` allowed) and compared as one
  string against the configured export. `..`, `.`, empty and NUL components are still
  refused. This is the only behavioural change to the ported wire core.
- **File ids are per-run.** The backend assigns ids lazily and encodes a per-process
  generation tag into each filehandle, so a handle from a previous run resolves to
  `NFS3ERR_STALE`. Model weights are immutable and clients remount per run, so this is
  sufficient; there is no on-disk id stability across restarts (wgshare got that from its
  manifest, which we dropped).
- **`READDIR` re-lists per entry.** The nfsd core asks the backend for one entry at a time
  (so each entry can carry a resume cookie), and the local backend re-reads + sorts the
  directory on each call — O(n²) syscalls per `READDIR` RPC for a directory of n entries.
  Fine for model repos (a handful of files per directory); revisit if a directory ever
  holds thousands of entries.
- **Open file handles are cached without eviction.** One `*os.File` per file read is kept
  open for the process lifetime. Model repos have few files; add an LRU if that ever stops
  being true.
- **No RDMA, no UDP, no NFSv4, no NLM/statd/quota.** Deliberate — see the design doc.
- **Not exercised against a real kernel NFS client in this repo's tests** (mounting needs
  root). Correctness is covered by over-the-wire ONC-RPC tests here and by wgshare's
  upstream fuzz/conformance suite for the ported core.
- On a timeout, the abandoned physical read's goroutine finishes on its own when the
  underlying `ReadAt` finally returns; there is no way to cancel a blocking `os.File.ReadAt`
  mid-syscall, so this is a bounded, transient goroutine rather than a leak.
