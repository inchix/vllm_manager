"""Unit tests for the layered config model + per-node overrides (docs/07).

Hardware-free. Run: `python3 -m pytest admin/cluster/test_config.py` or standalone.
"""
from __future__ import annotations

import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from admin.cluster import config as cfgmod  # noqa: E402


# COVID-like detected inventory (from admin/cluster/detect.detect_node()).
COVID_DETECTED = {
    "gpus": [{"index": 0, "name": "V100", "mem_total_mb": 32768}],
    "nvlink_active": False, "iommu": True,
    "rdma": [{"hca": "mlx4_0", "ports": [1, 2], "gid_index": 3, "netdev": "enp196s0",
              "fabric_ip": "172.16.254.201"}],
    "mgmt_ip": "172.16.69.201",
    "derived": {"nccl_ib_hca": "mlx4_0:1", "nccl_p2p_disable": True,
                "disable_custom_all_reduce": True, "nccl_socket_ifname": "enp196s0",
                "gloo_socket_ifname": "enp196s0"},
}


def test_image_defaults_present():
    eff = cfgmod.resolve_effective(None, None, None)
    assert eff["gpu_memory_utilization"] == 0.85
    assert eff["enforce_eager"] is True
    assert eff["multigpu_executor"] == "mp"


def test_detected_fills_fabric():
    eff = cfgmod.resolve_effective(COVID_DETECTED, None, None)
    assert eff["nccl_ib_hca"] == "mlx4_0:1"
    assert eff["nccl_p2p_disable"] is True
    assert eff["nccl_socket_ifname"] == "enp196s0"
    assert eff["ray_node_ip"] == "172.16.69.201"


def test_cluster_defaults_beat_image():
    eff = cfgmod.resolve_effective(None, {"gpu_memory_utilization": 0.70}, None)
    assert eff["gpu_memory_utilization"] == 0.70


def test_override_wins_over_everything():
    # override the detected NIC order and cluster util
    eff = cfgmod.resolve_effective(
        COVID_DETECTED, {"gpu_memory_utilization": 0.70},
        {"nccl_socket_ifname": "ens2,enp196s0", "gpu_memory_utilization": 0.60})
    assert eff["nccl_socket_ifname"] == "ens2,enp196s0"   # override beats detected
    assert eff["gpu_memory_utilization"] == 0.60          # override beats cluster default


def test_empty_override_does_not_mask_lower_layer():
    eff = cfgmod.resolve_effective(COVID_DETECTED, None, {"nccl_ib_hca": ""})
    assert eff["nccl_ib_hca"] == "mlx4_0:1"               # empty string ignored


def test_derived_gate_off_when_nvlink_present():
    det = dict(COVID_DETECTED)
    det["derived"] = {"nccl_p2p_disable": False, "disable_custom_all_reduce": False}
    eff = cfgmod.resolve_effective(det, None, None)
    assert eff["nccl_p2p_disable"] is False


# -- validation ---------------------------------------------------------------

def test_validate_rejects_unknown_key():
    errs = cfgmod.validate_override({"nonsense": 1})
    assert any("unknown setting" in e for e in errs)


def test_validate_gloo_single_iface():
    errs = cfgmod.validate_override({"gloo_socket_ifname": "ens2,enp196s0"})
    assert any("single interface" in e for e in errs)


def test_validate_util_range():
    assert cfgmod.validate_override({"gpu_memory_utilization": 0.85}) == []
    assert any("(0, 1]" in e for e in cfgmod.validate_override({"gpu_memory_utilization": 1.5}))


def test_validate_iface_must_be_detected():
    errs = cfgmod.validate_override({"gloo_socket_ifname": "eth9"}, COVID_DETECTED)
    assert any("not in detected NICs" in e for e in errs)


def test_validate_roles():
    assert any("unknown role" in e for e in cfgmod.validate_override({"roles": ["boss"]}))
    assert cfgmod.validate_override({"roles": ["admin", "participant"]}) == []


# -- store (persistence + snapshot) ------------------------------------------

def test_store_roundtrip_and_snapshot():
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "cluster-config.json")
    store = cfgmod.ClusterConfigStore(path)
    assert store.set_defaults({"gpu_memory_utilization": 0.80}) == []
    assert store.set_override("ebola", {"nccl_socket_ifname": "ens2,enp196s0"}) == []
    # effective merges detected + defaults + override
    eff = store.effective("ebola", COVID_DETECTED)
    assert eff["gpu_memory_utilization"] == 0.80
    assert eff["nccl_socket_ifname"] == "ens2,enp196s0"
    # persisted: a fresh store reads the same back
    store2 = cfgmod.ClusterConfigStore(path)
    assert store2.defaults()["gpu_memory_utilization"] == 0.80
    assert store2.override("ebola")["nccl_socket_ifname"] == "ens2,enp196s0"
    # snapshot carries detected/overrides/effective for the UI
    snap = store2.snapshot({"ebola": COVID_DETECTED})
    assert "ebola" in snap["nodes"]
    assert snap["nodes"]["ebola"]["effective"]["nccl_ib_hca"] == "mlx4_0:1"
    assert "settings" in snap


def test_store_rejects_invalid_override():
    tmp = tempfile.mkdtemp()
    store = cfgmod.ClusterConfigStore(os.path.join(tmp, "c.json"))
    errs = store.set_override("n1", {"gpu_memory_utilization": 5})
    assert errs and "n1" not in store._overrides


def _run_standalone():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} config tests passed")


if __name__ == "__main__":
    _run_standalone()
