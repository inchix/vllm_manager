"""agentd — the per-node CCP client.

Runs on every non-admin-only node (and on the admin box for its local participant/storage
roles). It detects local hardware, dials the admin's CCP WebSocket, registers, then streams
heartbeat + telemetry and executes admin commands via the Runner. On disconnect it reconnects
with jittered backoff and re-registers, so a node that reboots auto-rejoins (docs/02).

Run with:  python3 -m admin.agent.agentd
Config via env (see .env / docs/07):
  CCP_ADMIN_URL   ws(s)://<admin-host>:<port>   (else derived from ADMIN_HOST/RAY_HEAD_HOST+ADMIN_PORT)
  ADMIN_API_KEY   bearer key for the control channel
  ROLES           comma list: admin,participant,storage   (else mapped from ADMIN_ROLE)
  CLUSTER_ID, NODE_ID, CANONICAL_MODEL_PATH/MODELS_DIR
"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import socket
import ssl
import time
from typing import Optional

try:
    import websockets
except ImportError:  # pragma: no cover
    websockets = None

from ..cluster import protocol
from ..cluster.detect import detect_node
from .runner import Runner

log = logging.getLogger("admin.agent")
AGENT_VERSION = "0.4.0"


def _roles_from_env() -> list:
    raw = os.environ.get("ROLES")
    if raw:
        return [r.strip() for r in raw.split(",") if r.strip()]
    legacy = os.environ.get("ADMIN_ROLE", "").strip()
    return {
        "manager": ["admin", "participant", "storage"],
        "worker": ["participant"],
        "single": ["admin", "participant", "storage"],
    }.get(legacy, ["participant"])


def _admin_ws_url() -> str:
    url = os.environ.get("CCP_ADMIN_URL")
    if not url:
        host = os.environ.get("ADMIN_HOST") or os.environ.get("RAY_HEAD_HOST") or "127.0.0.1"
        port = os.environ.get("ADMIN_PORT", "7080")
        url = f"ws://{host}:{port}"
    url = url.replace("https://", "wss://").replace("http://", "ws://")
    if not url.startswith(("ws://", "wss://")):
        url = "ws://" + url
    return url.rstrip("/") + "/api/ccp"


class Agent:
    def __init__(self):
        self.node_id = os.environ.get("NODE_ID") or socket.gethostname()
        self.cluster_id = os.environ.get("CLUSTER_ID", "default")
        self.roles = protocol.validate_roles(_roles_from_env())
        self.api_key = os.environ.get("ADMIN_API_KEY", "")
        self.url = _admin_ws_url()
        self.canonical_model_path = (os.environ.get("CANONICAL_MODEL_PATH")
                                     or os.environ.get("MODELS_DIR", "/export/llm_models"))
        self.runner = Runner()
        self.detected = {}
        self._hb_sec = 5
        self._tel_sec = 10
        self._seq = 0
        self._started = time.time()
        self._stop = False

    def _register_body(self) -> dict:
        self.detected = detect_node()
        rdma = self.detected.get("rdma") or []
        fabric = [r.get("fabric_ip") for r in rdma if r.get("fabric_ip")]
        body = protocol.RegisterBody(
            node_id=self.node_id,
            hostname=socket.gethostname(),
            roles=self.roles,
            addresses={"mgmt": self.detected.get("mgmt_ip", ""), "fabric": fabric},
            gpus=[protocol.GpuInfo(index=g.get("index", 0), model=g.get("name", ""),
                                   mem_total_mb=g.get("mem_total_mb", 0), uuid=g.get("uuid", ""))
                  for g in self.detected.get("gpus", [])],
            rdma=[protocol.RdmaInfo(**{k: r.get(k) for k in
                                       ("hca", "ports", "gid_index", "netdev", "fabric_ip")})
                  for r in rdma],
            detected=self.detected,
            canonical_model_path=self.canonical_model_path,
            agent_version=AGENT_VERSION,
            cluster_id=self.cluster_id,
        )
        return body.to_dict()

    async def _connect(self):
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        ssl_ctx = None
        if self.url.startswith("wss://"):
            ssl_ctx = ssl.create_default_context()
            if os.environ.get("CCP_TLS_INSECURE") in ("1", "true", "yes"):
                ssl_ctx.check_hostname = False
                ssl_ctx.verify_mode = ssl.CERT_NONE
        # websockets renamed the header kwarg across versions
        for kw in ("additional_headers", "extra_headers"):
            try:
                return await websockets.connect(self.url, ssl=ssl_ctx, **{kw: headers},
                                                open_timeout=15, ping_interval=20)
            except TypeError:
                continue
        return await websockets.connect(self.url, ssl=ssl_ctx, open_timeout=15)

    async def run(self):
        if websockets is None:
            raise RuntimeError("the 'websockets' package is required for the agent")
        backoff = 1.0
        log.info("agent %s roles=%s -> %s", self.node_id, self.roles, self.url)
        while not self._stop:
            try:
                async with await self._connect() as ws:
                    await self._session(ws)
                    backoff = 1.0
            except Exception as exc:  # noqa: BLE001
                log.warning("connection lost (%s); retrying in %.0fs", exc, backoff)
            if self._stop:
                break
            await asyncio.sleep(backoff + random.uniform(0, 0.5))
            backoff = min(backoff * 2, 30.0)

    async def _session(self, ws):
        # register + await hello
        reg = protocol.make_frame(protocol.REGISTER, self._register_body())
        await ws.send(protocol.encode(reg))
        raw = await asyncio.wait_for(ws.recv(), timeout=20)
        hello = protocol.decode(raw)
        if hello["type"] == protocol.ERROR:
            raise RuntimeError(f"register rejected: {hello['body']}")
        self._apply_hello(hello["body"])
        log.info("registered; hb=%ss tel=%ss", self._hb_sec, self._tel_sec)

        tasks = [
            asyncio.ensure_future(self._heartbeat_loop(ws)),
            asyncio.ensure_future(self._telemetry_loop(ws)),
            asyncio.ensure_future(self._recv_loop(ws)),
        ]
        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        finally:
            for t in tasks:
                t.cancel()

    def _apply_hello(self, body: dict):
        self._hb_sec = int(body.get("heartbeat_sec", self._hb_sec))
        self._tel_sec = int(body.get("telemetry_sec", self._tel_sec))
        eff = body.get("effective_config") or {}
        try:
            self.runner.apply_config(eff)
        except Exception as exc:  # noqa: BLE001
            log.debug("apply_config: %s", exc)
        # reconcile desired replicas
        for rbody in (body.get("desired_state") or {}).get("replicas", []):
            try:
                self.runner.ensure_replica(rbody)
            except Exception as exc:  # noqa: BLE001
                log.warning("reconcile replica failed: %s", exc)

    async def _heartbeat_loop(self, ws):
        while True:
            self._seq += 1
            await ws.send(protocol.encode(protocol.heartbeat(self._seq, time.time() - self._started)))
            await asyncio.sleep(self._hb_sec)

    async def _telemetry_loop(self, ws):
        while True:
            frame = protocol.telemetry(self.runner.gpu_telemetry(),
                                       self.runner.replica_states(),
                                       self.runner.mount_states(),
                                       volumes=self.runner.volumes(),
                                       shares=self.runner.share_states())
            await ws.send(protocol.encode(frame))
            await asyncio.sleep(self._tel_sec)

    async def _recv_loop(self, ws):
        async for raw in ws:
            try:
                frame = protocol.decode(raw)
            except protocol.ProtocolError as exc:
                log.debug("bad frame: %s", exc)
                continue
            await self._handle_command(ws, frame)

    async def _handle_command(self, ws, frame: dict):
        ftype = frame["type"]
        body = frame["body"]
        rid = frame["id"]
        if ftype not in protocol.COMMAND_FRAMES and ftype not in (protocol.SET_CONFIG, protocol.BYE):
            return
        await ws.send(protocol.encode(protocol.ack(rid)))
        # dispatch to the runner in a thread so blocking subprocess work doesn't stall the loop
        loop = asyncio.get_event_loop()
        try:
            res = await loop.run_in_executor(None, self._exec_command, ftype, body)
        except Exception as exc:  # noqa: BLE001
            res = {"ok": False, "error": str(exc)}
        await ws.send(protocol.encode(protocol.result(
            rid, ok=bool(res.get("ok")), state=res.get("state", ""),
            detail=res.get("detail", ""), err=res.get("error", ""))))

    def _exec_command(self, ftype: str, body: dict) -> dict:
        r = self.runner
        if ftype == protocol.ENSURE_REPLICA:
            return r.ensure_replica(body)
        if ftype == protocol.STOP_REPLICA:
            return r.stop_replica(body["replica_id"], ray=body.get("ray", True))
        if ftype == protocol.MOUNT_STORAGE:
            return r.mount_storage(body)
        if ftype == protocol.UNMOUNT_STORAGE:
            return r.unmount_storage(body["canonical_path"])
        if ftype == protocol.SERVE_STORAGE:
            return r.serve_storage(body)
        if ftype == protocol.UNSHARE_STORAGE:
            return r.unshare_storage(body)
        if ftype == protocol.SYNC_MODEL:
            return r.sync_model(body)
        if ftype == protocol.SET_CONFIG:
            return r.apply_config(body.get("effective_config") or {})
        if ftype == protocol.BYE:
            self._stop = True
            return {"ok": True, "detail": "bye"}
        return {"ok": False, "error": f"unknown command {ftype}"}


def main():
    logging.basicConfig(
        level=os.environ.get("CCP_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    agent = Agent()
    try:
        asyncio.run(agent.run())
    except KeyboardInterrupt:
        agent.runner.shutdown()


if __name__ == "__main__":
    main()
