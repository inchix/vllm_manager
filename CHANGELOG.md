# Changelog

All notable changes to vLLM Manager are documented here.

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

Verified end to end: 4× V100 across 2 boxes (TP=2 intra-node × PP=2 inter-node)
serving inference over **RoCE RDMA**.

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
