"""CCP — the Cluster Control Protocol.

A tiny JSON-over-WebSocket protocol between a node **agent** and the **admin**. This module is
the shared wire contract: both sides import it. See docs/02-control-plane.md for the normative
spec.

Every message on the wire is one JSON object (a *frame*) with a fixed envelope::

    {"v": 1, "type": "<frame-type>", "id": "<uuid>", "ts": "<rfc3339>",
     "reply_to": "<uuid|null>", "body": { ... }}

The module is deliberately dependency-free (stdlib only) and Python 3.9+ compatible so the agent
can run anywhere the image runs.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

PROTOCOL_VERSION = 1

# ---------------------------------------------------------------------------
# Frame types
# ---------------------------------------------------------------------------

# agent -> admin
REGISTER = "register"
HEARTBEAT = "heartbeat"
TELEMETRY = "telemetry"
ACK = "ack"
RESULT = "result"
EVENT = "event"

# admin -> agent
HELLO = "hello"
SET_CONFIG = "set_config"
ENSURE_REPLICA = "ensure_replica"
STOP_REPLICA = "stop_replica"
MOUNT_STORAGE = "mount_storage"
UNMOUNT_STORAGE = "unmount_storage"
SERVE_STORAGE = "serve_storage"
UNSHARE_STORAGE = "unshare_storage"
SYNC_MODEL = "sync_model"
ERROR = "error"
BYE = "bye"

AGENT_FRAMES = {REGISTER, HEARTBEAT, TELEMETRY, ACK, RESULT, EVENT}
ADMIN_FRAMES = {
    HELLO, SET_CONFIG, ENSURE_REPLICA, STOP_REPLICA, MOUNT_STORAGE,
    UNMOUNT_STORAGE, SERVE_STORAGE, UNSHARE_STORAGE, SYNC_MODEL, ERROR, BYE,
}
# Commands the admin sends that expect a matching ack + result (by reply_to).
COMMAND_FRAMES = {
    ENSURE_REPLICA, STOP_REPLICA, MOUNT_STORAGE, UNMOUNT_STORAGE,
    SERVE_STORAGE, UNSHARE_STORAGE, SYNC_MODEL, SET_CONFIG,
}

# Roles
ROLE_ADMIN = "admin"
ROLE_PARTICIPANT = "participant"
ROLE_STORAGE = "storage"
ALL_ROLES = {ROLE_ADMIN, ROLE_PARTICIPANT, ROLE_STORAGE}

# Node lifecycle states (admin's view) — see docs/02.
STATE_READY = "READY"
STATE_SERVING = "SERVING"
STATE_DEGRADED = "DEGRADED"
STATE_DOWN = "DOWN"
STATE_DISCONNECTED = "DISCONNECTED"


# ---------------------------------------------------------------------------
# Envelope
# ---------------------------------------------------------------------------

def now_rfc3339() -> str:
    return datetime.now(timezone.utc).isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def make_frame(
    ftype: str,
    body: Optional[dict] = None,
    *,
    fid: Optional[str] = None,
    reply_to: Optional[str] = None,
) -> dict:
    """Build a well-formed frame dict."""
    return {
        "v": PROTOCOL_VERSION,
        "type": ftype,
        "id": fid or new_id(),
        "ts": now_rfc3339(),
        "reply_to": reply_to,
        "body": body or {},
    }


def encode(frame: dict) -> str:
    return json.dumps(frame, separators=(",", ":"))


class ProtocolError(ValueError):
    """A frame that violates the envelope contract."""


def decode(raw: str) -> dict:
    """Parse and structurally validate a wire frame. Raises ProtocolError on malformed input."""
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError) as exc:
        raise ProtocolError(f"not JSON: {exc}") from exc
    if not isinstance(obj, dict):
        raise ProtocolError("frame is not an object")
    if obj.get("v") != PROTOCOL_VERSION:
        raise ProtocolError(f"unsupported protocol version: {obj.get('v')!r}")
    ftype = obj.get("type")
    if not isinstance(ftype, str) or not ftype:
        raise ProtocolError("missing frame type")
    if not isinstance(obj.get("id"), str):
        raise ProtocolError("missing frame id")
    body = obj.get("body")
    if body is None:
        obj["body"] = {}
    elif not isinstance(body, dict):
        raise ProtocolError("body must be an object")
    obj.setdefault("reply_to", None)
    return obj


# ---------------------------------------------------------------------------
# Typed payload helpers (light dataclasses; everything serialises to plain dicts)
# ---------------------------------------------------------------------------

@dataclass
class GpuInfo:
    index: int
    model: str = ""
    mem_total_mb: int = 0
    uuid: str = ""
    # live fields (present in telemetry, usually absent in register)
    util: Optional[float] = None
    mem_used_mb: Optional[int] = None
    temp: Optional[float] = None
    power_draw: Optional[float] = None
    power_limit: Optional[float] = None

    def to_dict(self) -> dict:
        return {k: v for k, v in asdict(self).items() if v is not None}


@dataclass
class RdmaInfo:
    hca: str
    ports: list = field(default_factory=list)
    gid_index: Optional[int] = None
    netdev: str = ""
    fabric_ip: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class RegisterBody:
    node_id: str
    hostname: str
    roles: list
    addresses: dict                      # {"mgmt": ip, "fabric": [ip, ...]}
    gpus: list = field(default_factory=list)         # list[GpuInfo|dict]
    rdma: list = field(default_factory=list)         # list[RdmaInfo|dict]
    detected: dict = field(default_factory=dict)     # raw detect_node() output
    canonical_model_path: str = ""
    agent_version: str = ""
    cluster_id: str = "default"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["gpus"] = [g.to_dict() if isinstance(g, GpuInfo) else g for g in self.gpus]
        d["rdma"] = [r.to_dict() if isinstance(r, RdmaInfo) else r for r in self.rdma]
        return d


def validate_roles(roles) -> list:
    """Return a clean, ordered list of valid roles or raise ProtocolError."""
    if not isinstance(roles, (list, tuple)) or not roles:
        raise ProtocolError("roles must be a non-empty list")
    bad = [r for r in roles if r not in ALL_ROLES]
    if bad:
        raise ProtocolError(f"unknown role(s): {bad}")
    # stable canonical order
    order = [ROLE_ADMIN, ROLE_PARTICIPANT, ROLE_STORAGE]
    return [r for r in order if r in roles]


# Convenience constructors for the common admin->agent commands ---------------

def hello(effective_config: dict, desired_state: dict, *,
          heartbeat_sec: int, telemetry_sec: int, reply_to: Optional[str] = None) -> dict:
    return make_frame(HELLO, {
        "heartbeat_sec": heartbeat_sec,
        "telemetry_sec": telemetry_sec,
        "effective_config": effective_config,
        "desired_state": desired_state,
    }, reply_to=reply_to)


def error(code: str, detail: str = "", *, reply_to: Optional[str] = None) -> dict:
    return make_frame(ERROR, {"code": code, "detail": detail}, reply_to=reply_to)


def ack(reply_to: str, accepted: bool = True, note: str = "") -> dict:
    return make_frame(ACK, {"accepted": accepted, "note": note}, reply_to=reply_to)


def result(reply_to: str, ok: bool, state: str = "", detail: str = "",
           err: str = "") -> dict:
    return make_frame(RESULT, {"ok": ok, "state": state, "detail": detail, "error": err},
                      reply_to=reply_to)


def heartbeat(seq: int, uptime: float, load: Optional[list] = None) -> dict:
    return make_frame(HEARTBEAT, {"seq": seq, "uptime": uptime, "load": load or []})


def telemetry(gpus: list, replicas: list, mounts: list,
              volumes: Optional[list] = None, shares: Optional[list] = None) -> dict:
    return make_frame(TELEMETRY, {
        "gpus": [g.to_dict() if isinstance(g, GpuInfo) else g for g in gpus],
        "replicas": replicas,
        "mounts": mounts,
        "volumes": volumes or [],   # share candidates (storage role)
        "shares": shares or [],     # currently exported: {path, endpoint, ok}
    })
