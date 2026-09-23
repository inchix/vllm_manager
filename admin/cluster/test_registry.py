"""Unit tests for the node registry lifecycle — the auto-rejoin/recovery brain.

Deterministic, hardware-free. Validates the state machine that makes an ebola reboot self-heal:
register -> READY, heartbeat loss -> DOWN, reconnect -> READY.

Run: `python3 -m pytest admin/cluster/test_registry.py` or `python3 admin/cluster/test_registry.py`.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from admin.cluster import protocol  # noqa: E402
from admin.cluster.registry import Registry  # noqa: E402


def _body(nid="n1", roles=None, gpus=2):
    return {
        "node_id": nid,
        "hostname": nid,
        "roles": roles or ["participant", "storage"],
        "addresses": {"mgmt": "10.0.0.2", "fabric": ["172.16.254.2"]},
        "gpus": [{"index": i, "model": "V100", "mem_total_mb": 32768} for i in range(gpus)],
        "detected": {"derived": {"nccl_ib_hca": "mlx4_0:1"}},
    }


def test_register_ready_and_roles_ordered():
    r = Registry()
    node = r.register(_body(roles=["storage", "participant"]))
    assert node.state == protocol.STATE_READY
    assert node.connected is True
    # canonical role order: admin, participant, storage
    assert node.roles == ["participant", "storage"]


def test_heartbeat_loss_marks_down():
    r = Registry()
    r.register(_body("ebola"))
    r.get("ebola").last_heartbeat = time.time() - 100    # simulate 100s silence
    down = r.sweep(miss_seconds=15)
    assert "ebola" in down
    assert r.get("ebola").state == protocol.STATE_DOWN
    assert r.get("ebola").connected is False
    # sweeping again does not re-report an already-DOWN node
    assert r.sweep(miss_seconds=15) == []


def test_reboot_then_reconnect_auto_rejoins():
    r = Registry()
    r.register(_body("ebola"))
    # it dies
    r.get("ebola").last_heartbeat = time.time() - 100
    r.sweep(miss_seconds=15)
    assert r.get("ebola").state == protocol.STATE_DOWN
    # it reboots and its agent re-registers -> back to READY, no human involved
    r.register(_body("ebola"))
    assert r.get("ebola").state == protocol.STATE_READY
    assert r.get("ebola").connected is True


def test_disconnect_then_sweep_to_down():
    r = Registry()
    r.register(_body("n1"))
    r.mark_disconnected("n1")
    assert r.get("n1").state == protocol.STATE_DISCONNECTED
    r.get("n1").disconnected_at = time.time() - 100
    r.sweep(miss_seconds=15)
    assert r.get("n1").state == protocol.STATE_DOWN


def test_telemetry_failed_replica_degrades():
    r = Registry()
    r.register(_body("n1"))
    r.on_telemetry("n1", gpus=[], replicas=[{"id": "r1", "state": "FAILED"}], mounts=[])
    assert r.get("n1").state == protocol.STATE_DEGRADED
    # a healthy serving replica -> SERVING
    r.on_telemetry("n1", gpus=[], replicas=[{"id": "r1", "state": "SERVING"}], mounts=[])
    assert r.get("n1").state == protocol.STATE_SERVING


def test_participants_and_storage_filter_out_down():
    r = Registry()
    r.register(_body("gpu-a", roles=["participant"]))
    r.register(_body("store", roles=["storage"]))
    assert {n.node_id for n in r.participants()} == {"gpu-a"}
    assert {n.node_id for n in r.storage_nodes()} == {"store"}
    # take gpu-a down -> excluded from alive participants
    r.get("gpu-a").last_heartbeat = time.time() - 100
    r.sweep(miss_seconds=15)
    assert r.participants(alive_only=True) == []
    assert len(r.participants(alive_only=False)) == 1


def test_public_merges_live_telemetry():
    r = Registry()
    r.register(_body("n1"))
    r.on_telemetry("n1", gpus=[{"index": 0, "util": 55.0, "mem_used": 14903, "temp": 48.0}],
                   replicas=[], mounts=[])
    pub = r.public()[0]
    g0 = next(g for g in pub["gpus"] if g["index"] == 0)
    assert g0["model"] == "V100"          # base inventory
    assert g0["util"] == 55.0             # merged live telemetry
    assert g0["mem_used"] == 14903


def _run_standalone():
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"PASS {fn.__name__}")
    print(f"\n{len(fns)}/{len(fns)} registry tests passed")


if __name__ == "__main__":
    _run_standalone()
