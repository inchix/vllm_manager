"""Layered cluster configuration with per-node overrides.

Effective config for a node is a merge of four layers, lowest precedence first
(see docs/07-configuration.md)::

    1. image defaults       (IMAGE_DEFAULTS below)
    2. node-detected        (from admin/cluster/detect.detect_node(), mapped to config keys)
    3. cluster defaults     (admin, applies to all nodes)
    4. per-node override    (admin, per worker)  -- highest precedence

Only stdlib; Python 3.9+.
"""
from __future__ import annotations

import json
import os
import tempfile
import threading
from typing import Any, Optional

from . import protocol

# ---------------------------------------------------------------------------
# Known settings. `scope` documents where a key naturally lives; "node" keys are
# the override surface. Everything here is editable in the UI (docs/07).
# ---------------------------------------------------------------------------

# key -> {"scope": "cluster"|"node", "type": str, "help": str}
SETTINGS: dict = {
    # identity / roles (per-node)
    "roles": {"scope": "node", "type": "roles", "help": "admin/participant/storage set"},
    "cluster_id": {"scope": "node", "type": "str", "help": "cluster this node joins"},
    "ray_node_ip": {"scope": "node", "type": "str", "group": "network", "widget": "ip",
                    "label": "Cluster / control network",
                    "help": "NIC carrying CCP + Ray control traffic"},
    "ray_head_host": {"scope": "node", "type": "str", "help": "manager (Ray head) IP; workers"},
    "ray_head_port": {"scope": "cluster", "type": "int", "help": "Ray GCS port"},
    # fabric / NCCL (per-node — the heterogeneity that matters)
    "nccl_socket_ifname": {"scope": "node", "type": "csv", "group": "network",
                           "widget": "nic-list", "label": "Compute (NCCL) network",
                           "help": "NIC carrying tensor traffic; local NIC first"},
    "gloo_socket_ifname": {"scope": "node", "type": "iface", "group": "network",
                           "widget": "nic", "label": "Compute (Gloo) NIC",
                           "help": "single local NIC — Gloo cannot take a list"},
    "nccl_ib_hca": {"scope": "node", "type": "str", "help": "e.g. mlx4_0:1"},
    "nccl_ib_gid_index": {"scope": "node", "type": "int", "help": "RoCEv2 GID index"},
    "nccl_ib_disable": {"scope": "node", "type": "bool", "help": "disable IB/RoCE transport"},
    "nccl_p2p_disable": {"scope": "node", "type": "bool", "help": "disable PCIe P2P"},
    # execution / vLLM (mostly cluster-wide)
    "gpu_memory_utilization": {"scope": "cluster", "type": "float", "help": "0..1"},
    "multigpu_executor": {"scope": "cluster", "type": "str", "help": "mp|ray (single-node)"},
    "disable_custom_all_reduce": {"scope": "node", "type": "bool", "help": "route reductions via NCCL"},
    "enforce_eager": {"scope": "cluster", "type": "bool", "help": "skip CUDA graph capture (Volta cluster)"},
    "ray_cgraph_timeout": {"scope": "cluster", "type": "int",
                           "help": "seconds Ray Compiled Graph waits for a PP stage transfer "
                                   "(default 10 is too low for a large model in eager mode on "
                                   "slow GPUs -> RayChannelTimeoutError kills the engine)"},
    "dtype": {"scope": "cluster", "type": "str", "help": "auto|bfloat16|..."},
    "max_model_len": {"scope": "cluster", "type": "int", "help": "context length"},
    "port_start": {"scope": "cluster", "type": "int", "help": "vLLM port range start"},
    "port_end": {"scope": "cluster", "type": "int", "help": "vLLM port range end"},
    "cache_dir": {"scope": "cluster", "type": "str", "help": "compile cache dir (in-container)"},
    # storage
    "storage_shares": {"scope": "node", "type": "paths",
                       "help": "directories this node exports read-only (user-selectable)"},
    "storage_export_dir": {"scope": "node", "type": "str", "help": "modelfsd export dir"},
    "storage_listen": {"scope": "node", "type": "str", "help": "modelfsd fabric ip:port"},
    "storage_bind_ip": {"scope": "node", "type": "str", "group": "network", "widget": "ip",
                        "label": "Storage network",
                        "help": "NIC that serves NFS shares; the port is ephemeral and "
                                "advertised to the cluster"},
    "storage_allow": {"scope": "node", "type": "csv", "help": "client CIDRs"},
    "storage_readahead": {"scope": "node", "type": "str", "help": "e.g. 8m"},
    "storage_port_base": {"scope": "node", "type": "int",
                          "help": "first port for modelfsd shares (avoid 2049 if kernel "
                                  "nfsd is running); each share takes the next free port"},
    "canonical_model_path": {"scope": "cluster", "type": "str", "help": "same path on every node"},
    "mount_opts": {"scope": "cluster", "type": "csv", "help": "nfs mount options"},
    "storage_transport": {"scope": "cluster", "type": "str", "help": "tcp|rdma"},
    # hardware guards (per-node)
    "gpu_power_cap_w": {"scope": "node", "type": "int", "help": "nvidia-smi -pl watts; 0=default"},
    "gpu_persistence_mode": {"scope": "node", "type": "bool", "help": "nvidia-smi -pm 1"},
    # control plane (cluster-wide)
    "ccp_heartbeat_sec": {"scope": "cluster", "type": "int", "help": "agent heartbeat interval"},
    "ccp_telemetry_sec": {"scope": "cluster", "type": "int", "help": "telemetry interval"},
    "ccp_heartbeat_miss": {"scope": "cluster", "type": "int", "help": "misses before DOWN"},
}

IMAGE_DEFAULTS: dict = {
    "cluster_id": "default",
    "ray_head_port": 6379,
    "gpu_memory_utilization": 0.85,
    "multigpu_executor": "mp",
    "disable_custom_all_reduce": False,
    "enforce_eager": True,          # required for cluster launches on Volta (docs/04)
    # Ray Compiled Graph's default 10s read timeout kills PP transfers on slow GPUs.
    "ray_cgraph_timeout": 600,
    "dtype": "auto",
    "port_start": 8001,
    "port_end": 8010,
    "cache_dir": "/models/.vllm-cache",
    # The path INSIDE the container, where run.sh mounts MODELS_DIR. Agents report
    # their own value (CANONICAL_MODEL_PATH) at register and that wins; this is just
    # the fallback. It must not be the host path — the agent cannot see that.
    "canonical_model_path": "/models",
    "mount_opts": ["vers=3", "proto=tcp", "ro", "nofail", "soft", "timeo=100", "retrans=3"],
    "storage_transport": "tcp",
    "storage_readahead": "8m",
    # 2049 is the standard NFS port and is usually taken by the host's kernel nfsd,
    # so modelfsd defaults above it. Each additional share takes the next free port.
    "storage_port_base": 12049,
    "nccl_ib_disable": False,
    "gpu_power_cap_w": 0,
    "gpu_persistence_mode": False,
    "ccp_heartbeat_sec": 5,
    "ccp_telemetry_sec": 10,
    "ccp_heartbeat_miss": 3,
}


def derived_from_detected(detected: Optional[dict]) -> dict:
    """Map detect_node() output (docs/07 'node-detected') onto config keys."""
    if not detected:
        return {}
    d = detected.get("derived") or {}
    out: dict = {}
    for src, dst in (
        ("nccl_ib_hca", "nccl_ib_hca"),
        ("nccl_p2p_disable", "nccl_p2p_disable"),
        ("disable_custom_all_reduce", "disable_custom_all_reduce"),
        ("nccl_socket_ifname", "nccl_socket_ifname"),
        ("gloo_socket_ifname", "gloo_socket_ifname"),
    ):
        if d.get(src) is not None:
            out[dst] = d[src]
    if detected.get("mgmt_ip"):
        out["ray_node_ip"] = detected["mgmt_ip"]
    return out


def _clean(layer: Optional[dict]) -> dict:
    """Drop None/empty-string values so an empty override never masks a lower layer."""
    if not layer:
        return {}
    return {k: v for k, v in layer.items() if v is not None and v != ""}


def resolve_effective(detected: Optional[dict], cluster_defaults: Optional[dict],
                      node_override: Optional[dict]) -> dict:
    """Merge the four layers into the effective config for one node."""
    eff: dict = {}
    eff.update(IMAGE_DEFAULTS)
    eff.update(derived_from_detected(detected))
    eff.update(_clean(cluster_defaults))
    eff.update(_clean(node_override))
    return eff


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_override(override: dict, detected: Optional[dict] = None) -> list:
    """Return a list of human-readable errors for a per-node override (empty == ok)."""
    errs: list = []
    if not isinstance(override, dict):
        return ["override must be an object"]
    detected = detected or {}
    detected_ifaces = {r.get("netdev") for r in (detected.get("rdma") or []) if r.get("netdev")}
    for key, val in override.items():
        spec = SETTINGS.get(key)
        if spec is None:
            errs.append(f"unknown setting: {key}")
            continue
        t = spec["type"]
        try:
            if t == "roles":
                protocol.validate_roles(val)
            elif t == "iface":
                if not isinstance(val, str) or "," in val:
                    errs.append(f"{key}: must be a single interface (no comma)")
                elif detected_ifaces and val not in detected_ifaces:
                    errs.append(f"{key}: '{val}' not in detected NICs {sorted(detected_ifaces)} "
                                f"(pin explicitly if intentional)")
            elif t == "paths":
                if not isinstance(val, (list, tuple)):
                    errs.append(f"{key}: must be a list of absolute paths")
                else:
                    bad = [p for p in val if not isinstance(p, str) or not p.startswith("/")]
                    if bad:
                        errs.append(f"{key}: not absolute path(s): {bad}")
            elif t == "float":
                f = float(val)
                if key == "gpu_memory_utilization" and not (0.0 < f <= 1.0):
                    errs.append(f"{key}: must be in (0, 1]")
            elif t == "int":
                int(val)
            elif t == "bool":
                if not isinstance(val, bool):
                    errs.append(f"{key}: must be true/false")
        except (ValueError, TypeError, protocol.ProtocolError) as exc:
            errs.append(f"{key}: {exc}")
    return errs


# ---------------------------------------------------------------------------
# Persistent store: cluster defaults + per-node overrides
# ---------------------------------------------------------------------------

class ClusterConfigStore:
    """Holds cluster defaults + per-node overrides; persists to a JSON file.

    Thread-safe for the simple read/replace operations the admin performs.
    """

    def __init__(self, path: str):
        self.path = path
        self._lock = threading.RLock()
        self._defaults: dict = {}
        self._overrides: dict = {}   # node_id -> override dict
        self._load()

    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            self._defaults = data.get("defaults", {}) or {}
            self._overrides = data.get("nodes", {}) or {}
        except (FileNotFoundError, ValueError, OSError):
            self._defaults, self._overrides = {}, {}

    def _save(self) -> None:
        d = {"version": 1, "defaults": self._defaults, "nodes": self._overrides}
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=os.path.dirname(self.path) or ".", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(d, fh, indent=2)
            os.replace(tmp, self.path)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)

    # --- reads -----------------------------------------------------------
    def defaults(self) -> dict:
        with self._lock:
            return dict(self._defaults)

    def override(self, node_id: str) -> dict:
        with self._lock:
            return dict(self._overrides.get(node_id, {}))

    def effective(self, node_id: str, detected: Optional[dict]) -> dict:
        with self._lock:
            return resolve_effective(detected, self._defaults, self._overrides.get(node_id))

    # --- writes ----------------------------------------------------------
    def set_defaults(self, defaults: dict) -> list:
        errs = validate_override(defaults)  # same key/type checks
        if errs:
            return errs
        with self._lock:
            self._defaults = _clean(defaults)
            self._save()
        return []

    def set_override(self, node_id: str, override: dict, detected: Optional[dict] = None) -> list:
        errs = validate_override(override, detected)
        if errs:
            return errs
        with self._lock:
            cleaned = _clean(override)
            if cleaned:
                self._overrides[node_id] = cleaned
            else:
                self._overrides.pop(node_id, None)
            self._save()
        return []

    def snapshot(self, detected_by_node: Optional[dict] = None) -> dict:
        """Full config view for the UI: defaults + per-node {detected,overrides,effective}."""
        detected_by_node = detected_by_node or {}
        with self._lock:
            nodes = {}
            node_ids = set(self._overrides) | set(detected_by_node)
            for nid in node_ids:
                det = detected_by_node.get(nid)
                nodes[nid] = {
                    "detected": det or {},
                    "overrides": dict(self._overrides.get(nid, {})),
                    "effective": resolve_effective(det, self._defaults, self._overrides.get(nid)),
                }
            return {"defaults": dict(self._defaults), "nodes": nodes, "settings": SETTINGS}
