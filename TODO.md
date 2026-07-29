# TODO

## High Priority

- [x] HuggingFace token support for gated/private models (pass `HF_TOKEN` env var)
- [ ] Concurrent model downloads (currently limited to one at a time)
- [x] Auto-restart instances on crash (configurable, exponential backoff)
- [x] Persist instance configurations across container restarts

## Features

- [x] Model deletion from the UI
- [x] **Single-node multi-GPU (TP>1) now works** on this V100 hardware. Root cause of the long "hang" saga: (1) `NCCL_P2P_DISABLE=1` needed in all modes (broken PCIe P2P: no NVLink + IOMMU); (2) use the **mp** executor, not ray, for single-node — vLLM's Ray executor hangs the TP forward here while mp serves in ~3s; (3) auto `--disable-custom-all-reduce` for multi-GPU (custom all-reduce uses broken P2P). Verified: TP=2 serves inference via the admin UI. Also: compile caches now persist on the models volume (Volta JIT-compiles slowly); use dense models (Qwen3-Next/VL archs are Volta-hostile).
- [x] **Multi-node / remote GPUs over RDMA WORKS** (`CLUSTER_MODE`, Ray head/worker, "Use remote GPUs" toggle). Dense model runs TP=2×PP=2 across both boxes and serves inference via the admin UI. Same fix set as single-node (P2P disable + `--disable-custom-all-reduce` unblocked the Ray executor too).
  - [x] Confirmed NCCL uses RoCE RDMA (IB xmit +5.3MB during inference vs ~7KB on the ethernet NIC). GPUDirect RDMA is NOT supported on ConnectX-3, so host-staged RDMA is the ceiling here (nvidia_peermem won't help).
  - [ ] Cluster instance stop can still leak the Ray placement group (restart head+workers to clear).
  - [x] Auto layer-partition across nodes with uneven memory (32GB vs 16GB) — probes per-node GPU mem over Ray, weights layers (driver-first to match vLLM PP order); verified 24 layers -> 16/8 for the 64GB/32GB split. Override via explicit pp_layer_partition.
  - [ ] **BLOCKER: cross-node NCCL collective hangs (one PP communicator).** Long debug session narrowed it down. FIXED along the way and baked into run.sh/.env: (1) `NCCL_P2P_DISABLE=1` — PCIe P2P is broken (no active NVLink + host AMD-Vi IOMMU in Translated mode), so NCCL hangs at comm init unless forced to SHM; (2) `NCCL_IB_HCA=mlx4_0:1` — ConnectX-3 has 2 ports on different subnets (254 vs 253); NCCL paired port1<->port2 across nodes and the RoCE QP transition failed (ibv_modify_qp errno 22). With those, all 4 GPUs load the model and most NCCL comms reach "Init COMPLETE", but one inter-node PP communicator stalls connecting NET channels — and it hangs the same way over pure TCP (`NCCL_IB_DISABLE=1`) and with `--disable-custom-all-reduce`, so it's not RoCE- or P2P-specific. Observed: `NCCL_SOCKET_IFNAME` prefix-matches BOTH RoCE ports (ens2 -> ens2+ens2d1), spreading channels over both subnets; NCCL 2.27 does NOT support `=exact` match. **UPDATE (2026-07-30): fabric proven good.** A minimal 4-rank torch.distributed repro (nccl_test.py / nccl_test2.py in MODELS_DIR) PASSES with our settings — world + TP(intra) + PP(cross-node) subgroups all do new_group/barrier/all_reduce (gloo CPU + nccl GPU) correctly. So NCCL/Gloo/RoCE/firewall/P2P/port config are all correct; the hang is **vLLM+Ray-specific**, not the network. NEXT: investigate vLLM's custom PyNcclCommunicator (py-spy hang site) vs torch NCCL (works); check Ray GPU placement with RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES=1 (two workers on same GPU?); search vLLM issues for V100+multinode+pynccl hang; try a different vLLM version.
  - [ ] Live-test end to end once the init deadlock is resolved (RoCE NCCL, PP across nodes)
  - [ ] Auto layer-partition across nodes with uneven memory (currently manual; auto is local-only)
  - [ ] Worker health/status surfaced in the UI beyond Ray node list
  - [ ] Cluster instance stop can leak the Ray placement group (GPUs stay reserved) if killed mid-init; needs explicit PG cleanup or a "reset cluster" action. Workaround: restart the manager (Ray head) + workers.
- [ ] Instance resource monitoring (GPU utilization, memory per instance)
- [ ] Configurable vLLM arguments per instance (allowlisted flags — `extra_args` in `/api/start`)
- [ ] Model search from HuggingFace Hub in the UI
- [ ] Instance naming (custom names instead of instance-1, instance-2)
- [x] API key / basic auth for the admin UI (`AUTH_ENABLED`, `ADMIN_API_KEY`)
- [ ] docker-compose.yml / podman-compose.yml
- [ ] HTTPS support for admin UI (use a reverse proxy for now)

## UI Improvements

- [x] Dark/light theme toggle
- [x] Log filtering and search
- [x] Log download/export
- [x] GPU utilization charts over time
- [x] Mobile-responsive layout improvements
- [x] Toast notifications instead of alert() dialogs
- [x] Confirmation dialog before stopping instances
- [x] Login page + Sign out

## Security & Ops

- [x] API-key auth (header + cookie) with `AUTH_ENABLED=false` escape hatch for isolated on-prem
- [x] Bind admin port to loopback on host by default (`ADMIN_BIND_HOST`)
- [x] Safe `.env` parsing (no shell execution)
- [x] Allowlist for `extra_args` passed to vLLM
- [x] Download state race condition (asyncio.Lock)
- [x] Container hardening (`no-new-privileges`, `--cap-drop=ALL`, SELinux-ready)
- [x] Container HEALTHCHECK
- [x] systemd `Restart=on-failure`
- [ ] Rate limiting on login + download endpoints

## Technical Debt

- [ ] Add unit tests for vllm_manager.py
- [ ] Add integration tests for API endpoints
- [ ] Structured JSON logging
