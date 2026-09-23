"""Node registry — the admin's live view of the cluster.

Tracks each node's identity, inventory, roles, connection, heartbeat, telemetry and lifecycle
state (docs/02). Designed to be driven from a single asyncio loop (the CCP hub), so it holds no
locks; all mutation happens in the event loop.
"""
from __future__ import annotations

import time
from typing import Optional

from . import protocol


class Node:
    def __init__(self, node_id: str):
        self.node_id = node_id
        self.hostname = ""
        self.roles: list = []
        self.addresses: dict = {}
        self.gpus: list = []            # base inventory (from register), list[dict]
        self.rdma: list = []
        self.detected: dict = {}
        self.canonical_model_path = ""
        self.agent_version = ""
        self.cluster_id = "default"
        # live
        self.connected = False
        self.state = protocol.STATE_DISCONNECTED
        self.first_seen = time.time()
        self.last_seen = self.first_seen
        self.last_heartbeat = 0.0
        self.last_seq = -1
        self.disconnected_at: Optional[float] = None
        self.telemetry_gpus: list = []  # live per-gpu stats (from telemetry)
        self.replicas: list = []
        self.mounts: list = []

    # -- derived views ----------------------------------------------------
    def merged_gpus(self) -> list:
        """Base inventory with live telemetry fields merged in by index."""
        live = {g.get("index"): g for g in self.telemetry_gpus}
        out = []
        for base in self.gpus:
            idx = base.get("index")
            merged = dict(base)
            if idx in live:
                for k in ("util", "mem_used", "mem_used_mb", "mem_total", "temp",
                          "power_draw", "power_limit"):
                    if live[idx].get(k) is not None:
                        merged[k] = live[idx][k]
            out.append(merged)
        # if register carried no gpus but telemetry did, fall back to live
        if not out and self.telemetry_gpus:
            out = list(self.telemetry_gpus)
        return out

    def to_public(self) -> dict:
        return {
            "node_id": self.node_id,
            "hostname": self.hostname,
            "roles": self.roles,
            "state": self.state,
            "connected": self.connected,
            "addresses": self.addresses,
            "gpus": self.merged_gpus(),
            "replicas": self.replicas,
            "mounts": self.mounts,
            "canonical_model_path": self.canonical_model_path,
            "agent_version": self.agent_version,
            "last_seen": self.last_seen,
            "uptime_seen": round(self.last_seen - self.first_seen, 1),
        }


class Registry:
    def __init__(self):
        self._nodes: dict = {}

    # -- lifecycle transitions -------------------------------------------
    def register(self, body: dict) -> Node:
        node_id = body["node_id"]
        node = self._nodes.get(node_id) or Node(node_id)
        node.hostname = body.get("hostname", node.hostname)
        node.roles = protocol.validate_roles(body.get("roles", node.roles or ["participant"]))
        node.addresses = body.get("addresses", node.addresses)
        node.gpus = body.get("gpus", node.gpus)
        node.rdma = body.get("rdma", node.rdma)
        node.detected = body.get("detected", node.detected)
        node.canonical_model_path = body.get("canonical_model_path", node.canonical_model_path)
        node.agent_version = body.get("agent_version", node.agent_version)
        node.cluster_id = body.get("cluster_id", node.cluster_id)
        node.connected = True
        node.disconnected_at = None
        node.last_seen = time.time()
        node.last_heartbeat = node.last_seen
        node.state = protocol.STATE_SERVING if node.replicas else protocol.STATE_READY
        self._nodes[node_id] = node
        return node

    def on_heartbeat(self, node_id: str, seq: int) -> None:
        node = self._nodes.get(node_id)
        if not node:
            return
        node.last_heartbeat = time.time()
        node.last_seen = node.last_heartbeat
        node.last_seq = seq
        node.connected = True
        if node.state in (protocol.STATE_DISCONNECTED, protocol.STATE_DOWN):
            node.state = protocol.STATE_SERVING if node.replicas else protocol.STATE_READY

    def on_telemetry(self, node_id: str, gpus: list, replicas: list, mounts: list) -> None:
        node = self._nodes.get(node_id)
        if not node:
            return
        node.telemetry_gpus = gpus or []
        node.replicas = replicas or []
        node.mounts = mounts or []
        node.last_seen = time.time()
        failed = any(r.get("state") in ("FAILED", "ERROR") for r in node.replicas)
        if node.connected:
            if failed:
                node.state = protocol.STATE_DEGRADED
            elif node.replicas:
                node.state = protocol.STATE_SERVING
            else:
                node.state = protocol.STATE_READY

    def mark_disconnected(self, node_id: str) -> None:
        node = self._nodes.get(node_id)
        if not node:
            return
        node.connected = False
        node.disconnected_at = time.time()
        node.state = protocol.STATE_DISCONNECTED

    def sweep(self, miss_seconds: float) -> list:
        """Promote stale/disconnected nodes to DOWN. Returns node_ids newly marked DOWN."""
        now = time.time()
        newly_down = []
        for node in self._nodes.values():
            if node.state == protocol.STATE_DOWN:
                continue
            gap = now - node.last_heartbeat if node.last_heartbeat else None
            stale = gap is not None and gap > miss_seconds
            disc = (not node.connected and node.disconnected_at is not None
                    and now - node.disconnected_at > miss_seconds)
            if stale or disc:
                node.state = protocol.STATE_DOWN
                node.connected = False
                newly_down.append(node.node_id)
        return newly_down

    # -- reads ------------------------------------------------------------
    def get(self, node_id: str) -> Optional[Node]:
        return self._nodes.get(node_id)

    def all(self) -> list:
        return list(self._nodes.values())

    def public(self) -> list:
        return [n.to_public() for n in self._nodes.values()]

    def detected_by_node(self) -> dict:
        return {nid: n.detected for nid, n in self._nodes.items()}

    def participants(self, alive_only: bool = True) -> list:
        out = []
        for n in self._nodes.values():
            if protocol.ROLE_PARTICIPANT not in n.roles:
                continue
            if alive_only and n.state in (protocol.STATE_DOWN, protocol.STATE_DISCONNECTED):
                continue
            out.append(n)
        return out

    def storage_nodes(self, alive_only: bool = True) -> list:
        out = []
        for n in self._nodes.values():
            if protocol.ROLE_STORAGE not in n.roles:
                continue
            if alive_only and n.state in (protocol.STATE_DOWN, protocol.STATE_DISCONNECTED):
                continue
            out.append(n)
        return out
