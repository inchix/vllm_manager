"""Unit tests for the scheduler: layout, memory-weighted PP partition, executor selection.

Hardware-free. Run with `python3 -m pytest admin/cluster/test_scheduler.py` or standalone
`python3 admin/cluster/test_scheduler.py`.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from admin.cluster import scheduler  # noqa: E402
from admin.cluster.registry import Node  # noqa: E402


def _node(nid, ngpu, mem_mb, fabric_ip, mgmt_ip=None):
    n = Node(nid)
    n.gpus = [{"index": i, "model": "V100", "mem_total_mb": mem_mb} for i in range(ngpu)]
    n.addresses = {"mgmt": mgmt_ip or fabric_ip, "fabric": [fabric_ip]}
    n.roles = ["participant"]
    return n


# -- memory_weighted_partition ------------------------------------------------

def test_partition_sums_and_is_memory_weighted():
    # 32GB : 16GB over 40 layers -> the classic ~27:13 split
    part = scheduler.memory_weighted_partition([32768, 16384], 40)
    assert sum(part) == 40
    assert part[0] > part[1]
    assert part == [27, 13]


def test_partition_equal_weights_even():
    part = scheduler.memory_weighted_partition([16000, 16000], 32)
    assert sum(part) == 32
    assert abs(part[0] - part[1]) <= 1


def test_partition_min_one_per_stage():
    part = scheduler.memory_weighted_partition([1000, 1000, 1000], 4)
    assert sum(part) == 4
    assert all(x >= 1 for x in part)


def test_partition_too_few_layers_raises():
    try:
        scheduler.memory_weighted_partition([1, 1, 1], 2)
    except scheduler.ScheduleError:
        return
    raise AssertionError("expected ScheduleError for layers < stages")


# -- compute_layout -----------------------------------------------------------

def test_layout_homogeneous_ok():
    nodes = [_node("a", 2, 32768, "10.0.0.2"), _node("b", 2, 16384, "10.0.0.3")]
    lay = scheduler.compute_layout(nodes)
    assert lay["tp"] == 2 and lay["pp"] == 2 and lay["world"] == 4


def test_layout_heterogeneous_gpu_counts_raises():
    nodes = [_node("a", 2, 32768, "10.0.0.2"), _node("b", 1, 16384, "10.0.0.3")]
    try:
        scheduler.compute_layout(nodes)
    except scheduler.ScheduleError:
        return
    raise AssertionError("expected ScheduleError for uneven GPU counts")


def test_layout_driver_first_ordering():
    nodes = [_node("a", 2, 32768, "10.0.0.9"), _node("b", 2, 32768, "10.0.0.2")]
    lay = scheduler.compute_layout(nodes, head_id="a")
    assert lay["ordered_ids"][0] == "a"          # head first
    # without a forced head, order is by fabric IP
    lay2 = scheduler.compute_layout(nodes)
    assert lay2["ordered_ids"] == ["b", "a"]     # 10.0.0.2 before 10.0.0.9


# -- build_instance_plan: executor + enforce-eager gating ----------------------

def _eff(**over):
    base = {"enforce_eager": True, "disable_custom_all_reduce": True,
            "gpu_memory_utilization": 0.85, "ray_head_port": 6379}
    base.update(over)
    return base


def test_single_node_uses_mp_and_no_enforce_eager():
    nodes = [_node("a", 2, 32768, "10.0.0.2")]
    plan = scheduler.build_instance_plan("r1", nodes, "m", {"a": _eff()}, port=8001)
    assert plan["layout"]["pp"] == 1
    head_body = plan["commands"][0][1]
    assert head_body["layout"]["executor"] == "mp"
    assert "--enforce-eager" not in head_body["vllm_args"]      # single-node: graphs ok
    assert "--disable-custom-all-reduce" in head_body["vllm_args"]  # still multi-GPU on V100


def test_multi_node_uses_ray_and_enforce_eager():
    nodes = [_node("a", 2, 32768, "10.0.0.2"), _node("b", 2, 16384, "10.0.0.3")]
    eff = {"a": _eff(), "b": _eff()}
    plan = scheduler.build_instance_plan("r2", nodes, "m", eff, port=8001, num_layers=40)
    assert plan["layout"]["pp"] == 2
    assert plan["pp_layer_partition"] == "27,13"
    head_body = next(b for nid, b in plan["commands"] if b["role_in_instance"] == "head")
    assert head_body["layout"]["executor"] == "ray"
    assert "--enforce-eager" in head_body["vllm_args"]          # cluster PP: required


def test_per_node_env_carries_fabric_settings():
    nodes = [_node("a", 2, 32768, "10.0.0.2"), _node("b", 2, 16384, "10.0.0.3")]
    eff = {
        "a": _eff(nccl_socket_ifname="enp196s0,ens2", nccl_ib_hca="mlx4_0:1", nccl_p2p_disable=True),
        "b": _eff(nccl_socket_ifname="ens2,enp196s0", nccl_ib_hca="mlx4_0:1", nccl_p2p_disable=True),
    }
    plan = scheduler.build_instance_plan("r3", nodes, "m", eff, port=8001, num_layers=40)
    envs = {nid: b["env"] for nid, b in plan["commands"]}
    assert envs["a"]["NCCL_SOCKET_IFNAME"] == "enp196s0,ens2"   # local-first per node
    assert envs["b"]["NCCL_SOCKET_IFNAME"] == "ens2,enp196s0"
    assert envs["a"]["NCCL_P2P_DISABLE"] == "1"


def _run_standalone():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(fns)} scheduler tests passed")


if __name__ == "__main__":
    _run_standalone()
