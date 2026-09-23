# Changelog

All notable changes to vLLM Manager are documented here.

## v0.4.0 (unreleased) — control plane & composable roles

**Design lift, documented in [`docs/`](docs/README.md); not yet implemented.** A re-architecture
of how vLLM Manager spans machines. The proven v0.3.0 *data plane* (Ray + NCCL over RoCE, shared
models over NFS) is kept; a real *control plane* goes on top.

- **Composable node roles** (`admin` / `participant` / `storage`) replace the single
  `ADMIN_ROLE`. One box can be all three; a cluster can split them freely.
  ([docs/01](docs/01-roles.md))
- **Cluster Control Protocol (CCP)** — a JSON-over-WebSocket command-and-control protocol
  between a per-node **agent** and the admin: registration, heartbeat + GPU telemetry, and typed
  commands (`ensure_replica`, `mount_storage`, `serve_storage`, `sync_model`, `stop_replica`).
  Outbound-only from workers (no inbound ports, no SSH). ([docs/02](docs/02-control-plane.md))
- **Automatic liveness & recovery** — a node that hard-resets (see the ebola power saga) is
  detected on heartbeat loss and **auto-rejoins on reconnect** via a declarative reconciliation
  loop, instead of a manual `run.sh` + relaunch.
- **`modelfsd`** — an embedded, userspace, **read-only NFSv3/TCP** daemon for model weights,
  co-locatable on any node. Protocol core to be ported from the fuzz-tested `wgshare/internal/nfsd`;
  solves both "can't run `nfsd` in a hardened container" and "kernel NFS client wedges in D-state
  when the server vanishes." Transport (TCP vs kernel-RDMA) decided by a benchmark gate.
  ([docs/03](docs/03-storage-modelfsd.md))
- **Cross-node GPU monitoring** falls out of the telemetry stream (GPUs grouped by host, live).
- **Opt-in and backward-compatible**: gated by `CONTROL_PLANE`; unset == exact v0.3.0 behaviour.
  Relationship to community PR #1 (`cluster-two-host-ray`) and the cherry-pick plan are in
  [docs/05](docs/05-migration.md); phased roadmap with hardware gates in [docs/06](docs/06-roadmap.md).

## v0.3.0

### Added — Multi-node / remote GPUs over RDMA

Run a single vLLM instance across GPUs on **multiple boxes**, joined into one Ray
cluster over an RDMA fabric (RoCE/InfiniBand).

- **Cluster mode** (`CLUSTER_MODE=true`): the container switches to host networking,
  passes through the RDMA verbs devices (`/dev/infiniband/*`), and sets the NCCL
  RoCE environment.
- **Roles** (`ADMIN_ROLE`): `manager` runs the admin UI + a Ray head; `worker` runs
  the same image as a pure Ray worker on extra GPU boxes (`entrypoint.sh`).
- **Backend**: `/api/cluster` reports joined Ray nodes and total GPUs; the launch
  path derives tensor-parallel-within-node × pipeline-parallel-across-nodes from the
  live cluster shape (`use_cluster` on `/api/start`).
- **UI**: a Cluster panel (nodes + GPU totals) and a **"Use remote GPUs"** toggle on
  the launch form (appears once ≥2 nodes join); cluster instances get a `⛓ remote`
  badge.
- **Automatic memory-weighted PP layer partition** across nodes: probes each node's
  total GPU memory over Ray and gives bigger-memory nodes proportionally more layers
  (ordered to match vLLM's driver-first PP-stage assignment), so a 32 GB + 16 GB
  cluster splits e.g. 2:1 instead of evenly and doesn't OOM the smaller GPUs. Override
  with an explicit `pp_layer_partition`.

Verified end to end: 4× V100 across 2 boxes (TP=2 intra-node × PP=2 inter-node)
serving inference over **RoCE RDMA**.

### Added — `setup.sh` (hardware-detecting installer, cluster-aware)

- `./setup.sh` auto-detects the local hardware — GPU count, active NVLink, IOMMU,
  RDMA HCA + port count, RoCEv2 GID index, RoCE NIC — and generates a `.env` with
  the right NCCL/executor settings (e.g. pins `mlx4_0:1` only when a card has
  multiple ports; disables P2P + custom all-reduce only when NVLink is absent and
  IOMMU is on). Nothing about any specific fabric is hardcoded.
- Handles `--role manager|worker|single`, `--head-host`, `--models-dir`, etc.,
  interactive or non-interactive (`--yes`), and offers to build + run.

### Changed — no hardcoded hardware/cluster specifics

Everything cluster- or hardware-specific is now configurable via `.env` with
neutral defaults (nothing baked into committed code):

- `NCCL_IB_HCA`, `NCCL_IB_GID_INDEX`, `NCCL_P2P_DISABLE` now default **empty**
  (NCCL auto-detects) and are only passed to the container when set — the fabric
  specifics (e.g. `mlx4_0:1`, GID `3`, P2P off) live in your `.env`.
- New `MULTIGPU_EXECUTOR` (default `mp`) and `DISABLE_CUSTOM_ALL_REDUCE`
  (default off) replace the hardcoded single-node executor choice and the
  always-on `--disable-custom-all-reduce` (both were V100/IOMMU workarounds).
- New `CACHE_DIR` to relocate/disable the persistent compile cache.
- `mistral_common>=1.11.5` added to the image (required by transformers 5.14's
  Mistral/Devstral tokenizer path; the base ships 1.9.1 → `NameError: SpecialTokens`).

### Added — reproducibility & caching

- **Base image pinned by digest** (not the floating `nightly`/`latest` tag) so every
  node in a cluster runs the byte-identical vLLM/NCCL build.
- **Persistent compile cache** on the models volume (`VLLM_CACHE_ROOT`,
  `TORCHINDUCTOR_CACHE_DIR`, `TRITON_CACHE_DIR`) — the rootfs is read-only and the
  default cache dirs are tmpfs, so without this every start recompiled kernels from
  scratch (pathologically slow on older GPUs).

### Fixed — Volta / V100 multi-GPU support

Getting multi-GPU (and multi-node) working on V100 (Volta, `sm_70`) required several
hardware-specific fixes, all now defaulted in `run.sh` / `vllm_manager.py`:

- **`NCCL_P2P_DISABLE=1` in all modes** (previously cluster-only). With no active
  NVLink and the host IOMMU in translated mode, GPU PCIe P2P is broken and NCCL hangs
  at comm-init on any TP>1 — forcing the SHM transport fixes it.
- **Executor: `mp` for single-node multi-GPU, `ray` only for multi-node.** vLLM's Ray
  executor hangs the tensor-parallel forward on this hardware; the multiprocessing
  executor profiles in seconds and serves.
- **Auto `--disable-custom-all-reduce` for multi-GPU.** vLLM's custom all-reduce uses
  the broken P2P and hangs; disabling it routes reductions through NCCL (SHM/NET) and
  also unblocks the Ray executor for multi-node.
- **`NCCL_IB_HCA=mlx4_0:1`** — ConnectX-3 exposes two ports on different subnets;
  pinning to one port avoids a cross-subnet RoCE queue-pair failure.
- **Heterogeneous NIC names**: `NCCL_SOCKET_IFNAME` takes a comma list with the local
  NIC first (Ray copies the driver's value to all workers); `GLOO_SOCKET_IFNAME`
  defaults to the first (local) entry since Gloo can't parse a list.
- Pinned the base image to a **CUDA-12 (Volta-compatible) vLLM build** — the current
  `:nightly` ships torch on CUDA 13, which dropped Volta and fails on V100 with
  "no kernel image is available for execution on the device".

> **Notes for this hardware:** GPUDirect RDMA is not supported on ConnectX-3, so NCCL
> stages RoCE RDMA through host memory (the expected, working path). Vision-language /
> Qwen3-Next model architectures use Triton/`torch.compile` paths that are slow or
> unsupported on Volta — prefer standard dense models. See `README.md` and `TODO.md`.

### Earlier v0.3.0 work

- Persist vLLM subprocess logs to disk; fix a `stop()` race.
- Bump `transformers` to `>=5.5,<6`.
- Install Ray in the image; rename port env vars.
- Require `tool_call_parser` when `enable_tool_use` is true.
- Whitelist `/healthz`; build podman images in docker format (preserve HEALTHCHECK).
