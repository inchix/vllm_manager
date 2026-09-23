"""Scheduler — turn a launch request into per-node CCP commands.

Given a set of live participant nodes and the effective config, compute the TP×PP layout
(TP within a node, PP across nodes), the memory-weighted pipeline layer partition, pick the head
node, allocate a port, and emit one `ensure_instance` body per node. The layout rules mirror the
v0.3.0 cluster launch path (docs/04): homogeneous TP per node, PP = node count, driver-first
ordering to match vLLM's `sort_by_driver_then_worker_ip`.
"""
from __future__ import annotations

from typing import Optional

from . import protocol


class ScheduleError(ValueError):
    pass


def _node_ip(node) -> str:
    fabric = (node.addresses or {}).get("fabric") or []
    if fabric:
        return fabric[0]
    return (node.addresses or {}).get("mgmt", "")


def _ray_ip(node, cfg=None) -> str:
    """Address Ray should register this node under (control plane, not the fabric)."""
    cfg = cfg or {}
    if cfg.get("ray_node_ip"):
        return str(cfg["ray_node_ip"])
    mgmt = (node.addresses or {}).get("mgmt")
    if mgmt:
        return mgmt
    return _node_ip(node)


def _gpu_count(node) -> int:
    return len(node.gpus) or len(node.telemetry_gpus)


def _node_mem_mb(node) -> int:
    """Effective capacity of a node for TP, in MB.

    NOT the sum: tensor parallelism shards each tensor EVENLY across ranks, so a
    TP group is bounded by its SMALLEST GPU — a node with a 32 GB and a 16 GB card
    behaves like 2x16 GB, not 48 GB. Summing would over-weight that node and hand
    it more pipeline layers than it can hold.
    """
    mems = [int(g.get("mem_total_mb") or 0) for g in node.gpus]
    mems = [m for m in mems if m > 0]
    if not mems:
        return 0
    return min(mems) * len(mems)


def order_nodes(nodes: list, head_id: Optional[str] = None) -> list:
    """Driver-first, then by fabric IP — matches vLLM PP stage assignment."""
    ordered = sorted(nodes, key=lambda n: _node_ip(n))
    if head_id:
        head = [n for n in ordered if n.node_id == head_id]
        rest = [n for n in ordered if n.node_id != head_id]
        ordered = head + rest
    return ordered


def memory_weighted_partition(mem_weights: list, num_layers: int) -> list:
    """Split `num_layers` across stages proportional to per-stage memory.

    Largest remainders get the leftover layers; every stage gets at least 1.
    `mem_weights` is already in stage (driver-first) order.
    """
    n = len(mem_weights)
    if n == 0:
        return []
    if num_layers < n:
        raise ScheduleError(f"{num_layers} layers < {n} stages")
    total = float(sum(mem_weights)) or float(n)
    raw = [num_layers * (w / total) for w in mem_weights]
    floors = [max(1, int(x)) for x in raw]
    # fix up so the sum equals num_layers exactly
    diff = num_layers - sum(floors)
    if diff > 0:
        # hand extra layers to the largest fractional remainders
        order = sorted(range(n), key=lambda i: raw[i] - int(raw[i]), reverse=True)
        for i in range(diff):
            floors[order[i % n]] += 1
    elif diff < 0:
        # too many (min-1 pushed us over): trim from the largest stages
        order = sorted(range(n), key=lambda i: floors[i], reverse=True)
        k = 0
        while diff < 0:
            i = order[k % n]
            if floors[i] > 1:
                floors[i] -= 1
                diff += 1
            k += 1
    return floors


def compute_layout(nodes: list, head_id: Optional[str] = None) -> dict:
    """Validate homogeneity and return the TP×PP layout for these participant nodes."""
    if not nodes:
        raise ScheduleError("no participant nodes selected")
    counts = {n.node_id: _gpu_count(n) for n in nodes}
    if any(c == 0 for c in counts.values()):
        raise ScheduleError(f"a selected node has no GPUs: {counts}")
    uniq = set(counts.values())
    if len(uniq) != 1:
        raise ScheduleError(
            f"heterogeneous GPU counts per node {counts}; vLLM needs uniform TP. "
            "Select nodes with equal GPU counts, or launch per-node."
        )
    tp = uniq.pop()
    ordered = order_nodes(nodes, head_id)
    pp = len(ordered)
    return {
        "tp": tp,
        "pp": pp,
        "world": tp * pp,
        "head_id": ordered[0].node_id,
        "ordered_ids": [n.node_id for n in ordered],
        "ordered_nodes": ordered,
    }


def build_instance_plan(
    instance_id: str,
    nodes: list,
    model: str,
    effective_by_node: dict,
    *,
    port: int,
    num_layers: Optional[int] = None,
    head_id: Optional[str] = None,
    served_model_name: Optional[str] = None,
    max_model_len: Optional[int] = None,
    extra_args: Optional[list] = None,
) -> dict:
    """Produce {instance_id, layout, head_id, commands:[(node_id, ensure_instance_body)]}."""
    layout = compute_layout(nodes, head_id)
    ordered = layout["ordered_nodes"]
    head = ordered[0]
    head_cfg = effective_by_node.get(head.node_id, {})

    pp_partition = None
    if layout["pp"] > 1 and num_layers:
        weights = [_node_mem_mb(n) for n in ordered]
        pp_partition = ",".join(str(x) for x in memory_weighted_partition(weights, num_layers))

    # Ray CONTROL traffic uses the management address, not the RDMA fabric. vLLM
    # derives its placement-group node affinity ("node:<ip>") from the node's primary
    # address, so if Ray registers nodes under fabric IPs the bundle can never be
    # satisfied ("No available node types can fulfill resource request"). NCCL still
    # rides the fabric via NCCL_SOCKET_IFNAME — this only affects Ray scheduling.
    ray = {"head_addr": _ray_ip(head, head_cfg),
           "port": int(head_cfg.get("ray_head_port", 6379))}
    # Executor: multi-node uses Ray; single-node multi-GPU uses mp (Ray hangs the TP forward on
    # this V100+IOMMU hardware — the v0.3.0 lesson, docs/04).
    executor = "ray" if layout["pp"] > 1 else head_cfg.get("multigpu_executor", "mp")
    multi_gpu = layout["world"] > 1

    def vllm_args_for() -> list:
        args = list(extra_args or [])
        # --enforce-eager is REQUIRED for the cluster PP path (CUDA-graph capture crashes on
        # V100+PP-over-RDMA); single-node graphs are fine, so only force it when pp>1.
        if layout["pp"] > 1 and head_cfg.get("enforce_eager", True) \
                and "--enforce-eager" not in args:
            args.append("--enforce-eager")
        # custom all-reduce uses the broken PCIe P2P on this hardware; disable for any multi-GPU.
        if multi_gpu and head_cfg.get("disable_custom_all_reduce") \
                and "--disable-custom-all-reduce" not in args:
            args.append("--disable-custom-all-reduce")
        return args

    commands = []
    for i, node in enumerate(ordered):
        body = {
            "instance_id": instance_id,
            "role_in_instance": "head" if i == 0 else "worker",
            "ray": ray,
            "model": model,
            "layout": {
                "tp": layout["tp"],
                "pp": layout["pp"],
                "pp_layer_partition": pp_partition,
                "executor": executor,
            },
            "port": port if i == 0 else None,
            "served_model_name": served_model_name,
            "max_model_len": max_model_len or head_cfg.get("max_model_len"),
            "gpu_memory_utilization": head_cfg.get("gpu_memory_utilization", 0.85),
            "dtype": head_cfg.get("dtype", "auto"),
            "vllm_args": vllm_args_for(),
            "env": _instance_env(effective_by_node.get(node.node_id, {})),
        }
        commands.append((node.node_id, body))

    return {
        "instance_id": instance_id,
        "layout": {k: layout[k] for k in ("tp", "pp", "world", "head_id", "ordered_ids")},
        "head_id": layout["head_id"],
        "port": port,
        "pp_layer_partition": pp_partition,
        "commands": commands,
    }


def _instance_env(cfg: dict) -> dict:
    """Per-node NCCL/fabric env for the instance, from that node's effective config."""
    env = {}
    mapping = {
        "nccl_socket_ifname": "NCCL_SOCKET_IFNAME",
        "gloo_socket_ifname": "GLOO_SOCKET_IFNAME",
        "nccl_ib_hca": "NCCL_IB_HCA",
        "nccl_ib_gid_index": "NCCL_IB_GID_INDEX",
        "ray_node_ip": "RAY_NODE_IP",
    }
    for key, envname in mapping.items():
        val = cfg.get(key)
        if val not in (None, ""):
            env[envname] = str(val)
    if cfg.get("nccl_p2p_disable"):
        env["NCCL_P2P_DISABLE"] = "1"
    if cfg.get("nccl_ib_disable"):
        env["NCCL_IB_DISABLE"] = "1"
    return env
