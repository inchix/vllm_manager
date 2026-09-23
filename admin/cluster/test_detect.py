"""Tests for admin/cluster/detect.py.

Hermetic: nothing here depends on real GPUs or RDMA being present. The
smoke test only asserts shape (so it passes on any box), and the derivation
tests exercise the pure ``derive()`` logic with synthetic inputs. Runnable
under pytest OR directly: ``python3 admin/cluster/test_detect.py``.
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detect  # noqa: E402


# ---------------------------------------------------------------------------
# smoke: detect_node() runs without raising and has the documented shape
# ---------------------------------------------------------------------------

def test_detect_node_runs_and_shape():
    node = detect.detect_node()
    assert isinstance(node, dict)
    for key in ("gpus", "nvlink_active", "iommu", "rdma", "mgmt_ip", "derived"):
        assert key in node, "missing top-level key: {}".format(key)

    assert isinstance(node["gpus"], list)
    assert isinstance(node["nvlink_active"], bool)
    assert isinstance(node["iommu"], bool)
    assert isinstance(node["rdma"], list)
    assert node["mgmt_ip"] is None or isinstance(node["mgmt_ip"], str)

    derived = node["derived"]
    assert isinstance(derived, dict)
    for key in (
        "nccl_ib_hca",
        "nccl_p2p_disable",
        "disable_custom_all_reduce",
        "nccl_socket_ifname",
        "gloo_socket_ifname",
    ):
        assert key in derived, "missing derived key: {}".format(key)
    assert isinstance(derived["nccl_p2p_disable"], bool)
    assert isinstance(derived["disable_custom_all_reduce"], bool)


def test_gpu_entries_well_formed():
    """Whatever GPUs are (or aren't) present, entries must be well-formed."""
    for g in detect.detect_node()["gpus"]:
        assert set(("index", "name", "mem_total_mb", "uuid")).issubset(g)
        assert isinstance(g["index"], int)


# ---------------------------------------------------------------------------
# derivation rules (pure logic, synthetic inputs)
# ---------------------------------------------------------------------------

def _rdma(hca="mlx4_0", ports=(1,), netdev="ens2"):
    return [{"hca": hca, "ports": list(ports), "gid_index": 3,
             "netdev": netdev, "fabric_ip": "172.16.254.230"}]


def test_p2p_off_when_nvlink_present_even_if_iommu():
    # NVLink present → P2P/DCAR stay enabled regardless of IOMMU.
    d = detect.derive(nvlink_active=True, iommu=True, rdma=_rdma())
    assert d["nccl_p2p_disable"] is False
    assert d["disable_custom_all_reduce"] is False


def test_p2p_on_only_when_no_nvlink_and_iommu():
    d = detect.derive(nvlink_active=False, iommu=True, rdma=_rdma())
    assert d["nccl_p2p_disable"] is True
    assert d["disable_custom_all_reduce"] is True


def test_p2p_off_when_no_nvlink_and_no_iommu():
    d = detect.derive(nvlink_active=False, iommu=False, rdma=_rdma())
    assert d["nccl_p2p_disable"] is False
    assert d["disable_custom_all_reduce"] is False


def test_hca_pin_includes_port_only_when_multiport():
    # Single-port card → no ":port" pin.
    single = detect.derive(False, False, _rdma(ports=(1,)))
    assert single["nccl_ib_hca"] == "mlx4_0"
    # Multi-port card → pin the first port.
    multi = detect.derive(False, False, _rdma(ports=(1, 2)))
    assert multi["nccl_ib_hca"] == "mlx4_0:1"


def test_socket_ifnames_follow_primary_netdev():
    d = detect.derive(False, False, _rdma(netdev="ens2"))
    assert d["nccl_socket_ifname"] == "ens2"
    assert d["gloo_socket_ifname"] == "ens2"


def test_derive_no_rdma_is_safe():
    d = detect.derive(False, True, rdma=[])
    assert d["nccl_ib_hca"] is None
    assert d["nccl_socket_ifname"] is None
    assert d["gloo_socket_ifname"] is None
    # P2P gate still applies without RDMA.
    assert d["nccl_p2p_disable"] is True


# ---------------------------------------------------------------------------
# gid matching (synthetic sysfs via monkeypatching the file/dir readers)
# ---------------------------------------------------------------------------

def test_gid_index_picks_first_roce_v2_ipv4(monkeypatch):
    gids = {
        "0": ("IB/RoCE v1", "fe80:0000:0000:0000:ee0d:9aff:fe06:f180"),
        "1": ("RoCE v2", "fe80:0000:0000:0000:ee0d:9aff:fe06:f180"),  # v2 but IPv6 link-local
        "2": ("IB/RoCE v1", "0000:0000:0000:0000:0000:ffff:ac10:fec9"),  # v4 but v1
        "3": ("RoCE v2", "0000:0000:0000:0000:0000:ffff:ac10:fec9"),   # <- expected
    }

    def fake_listdir(path):
        if path.endswith("/gids"):
            return list(gids.keys())
        return []

    def fake_read(path):
        idx = os.path.basename(path)
        if "/gid_attrs/types/" in path:
            return gids.get(idx, ("", ""))[0]
        if "/gids/" in path:
            return gids.get(idx, ("", ""))[1]
        return ""

    monkeypatch.setattr(detect, "_listdir", fake_listdir)
    monkeypatch.setattr(detect, "_read_file", fake_read)
    assert detect.detect_gid_index("mlx4_0", 1) == 3


def test_gid_index_none_when_no_v2(monkeypatch):
    monkeypatch.setattr(detect, "_listdir", lambda p: ["0"] if p.endswith("/gids") else [])
    monkeypatch.setattr(detect, "_read_file", lambda p: "IB/RoCE v1")
    assert detect.detect_gid_index("mlx4_0", 1) is None


# ---------------------------------------------------------------------------
# plain-script fallback (no pytest required)
# ---------------------------------------------------------------------------

class _MP:
    """Minimal monkeypatch stand-in for the no-pytest runner."""

    def __init__(self):
        self._undo = []

    def setattr(self, obj, name, value):
        self._undo.append((obj, name, getattr(obj, name)))
        setattr(obj, name, value)

    def undo(self):
        for obj, name, old in reversed(self._undo):
            setattr(obj, name, old)
        self._undo = []


def _main():
    failures = 0
    for name, fn in sorted(globals().items()):
        if not (name.startswith("test_") and callable(fn)):
            continue
        mp = _MP()
        try:
            if "monkeypatch" in fn.__code__.co_varnames[: fn.__code__.co_argcount]:
                fn(mp)
            else:
                fn()
            print("ok   - {}".format(name))
        except AssertionError as e:
            failures += 1
            print("FAIL - {}: {}".format(name, e))
        except Exception as e:  # noqa: BLE001
            failures += 1
            print("ERROR- {}: {!r}".format(name, e))
        finally:
            mp.undo()
    if failures:
        print("\n{} test(s) failed".format(failures))
        sys.exit(1)
    print("\nall tests passed")


if __name__ == "__main__":
    _main()
