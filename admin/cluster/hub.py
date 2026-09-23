"""CCP hub — the admin's control-plane server.

Provides a FastAPI router with:
  * ``WS  /api/ccp``            the Cluster Control Protocol endpoint (agents dial in here)
  * ``GET /api/cluster/nodes``  live node + GPU inventory (for the UI monitor)
  * ``GET /api/cluster/config`` layered config snapshot (defaults + per-node)
  * ``POST /api/cluster/config`` update defaults / per-node overrides, pushed to agents
  * ``GET /api/cluster/summary`` quick counts
  * ``POST /api/cluster/launch`` schedule a distributed replica across participants (Phase 2)
  * ``POST /api/cluster/stop``   stop a replica

The hub owns a Registry and a ClusterConfigStore, tracks one live WebSocket per node, and runs a
background sweep that promotes silent nodes to DOWN.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Optional

from fastapi import APIRouter, Header, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import protocol, scheduler
from .config import ClusterConfigStore
from .registry import Registry

log = logging.getLogger("admin.cluster.hub")


class ClusterHub:
    def __init__(self, config_path: str, api_key: str = "", auth_enabled: bool = True,
                 agent_version: str = "0.4.0"):
        self.registry = Registry()
        self.config = ClusterConfigStore(config_path)
        self.api_key = api_key
        self.auth_enabled = auth_enabled
        self.agent_version = agent_version
        self._conns: dict = {}                 # node_id -> WebSocket
        self._pending: dict = {}               # frame id -> asyncio.Future (result)
        self._replicas: dict = {}              # replica_id -> plan metadata
        self._sweep_task: Optional[asyncio.Task] = None
        self._stop = False

    # -- auth -------------------------------------------------------------
    def check_key(self, presented: Optional[str]) -> bool:
        if not self.auth_enabled:
            return True
        return bool(presented) and presented == self.api_key

    def _key_from_ws(self, ws: WebSocket) -> Optional[str]:
        auth = ws.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return ws.headers.get("x-api-key") or ws.query_params.get("key")

    # -- lifecycle --------------------------------------------------------
    def start(self) -> None:
        if self._sweep_task is None:
            self._stop = False
            self._sweep_task = asyncio.ensure_future(self._sweep_loop())

    async def stop(self) -> None:
        self._stop = True
        if self._sweep_task:
            self._sweep_task.cancel()
            try:
                await self._sweep_task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass
            self._sweep_task = None

    async def _sweep_loop(self) -> None:
        while not self._stop:
            try:
                miss = self.config.defaults().get("ccp_heartbeat_miss", 3)
                hb = self.config.defaults().get("ccp_heartbeat_sec", 5)
                newly_down = self.registry.sweep(miss_seconds=max(3, int(miss) * int(hb)))
                for nid in newly_down:
                    log.warning("node %s marked DOWN (heartbeat lost)", nid)
                    self._on_node_down(nid)
            except Exception as exc:  # noqa: BLE001
                log.debug("sweep error: %s", exc)
            await asyncio.sleep(2)

    def _on_node_down(self, node_id: str):
        # mark any replica that depends on this node as failed (hold+alert policy, docs/06)
        for rid, meta in self._replicas.items():
            if node_id in meta.get("node_ids", []) and meta.get("state") != "FAILED":
                meta["state"] = "FAILED"
                meta["detail"] = f"node {node_id} went down"

    # -- send / command ---------------------------------------------------
    async def _send(self, node_id: str, frame: dict) -> bool:
        ws = self._conns.get(node_id)
        if ws is None:
            return False
        try:
            await ws.send_text(protocol.encode(frame))
            return True
        except Exception as exc:  # noqa: BLE001
            log.debug("send to %s failed: %s", node_id, exc)
            return False

    async def send_command(self, node_id: str, frame: dict, timeout: float = 120.0) -> dict:
        """Send a command and await its `result` frame (matched by reply_to)."""
        loop = asyncio.get_event_loop()
        fut = loop.create_future()
        self._pending[frame["id"]] = fut
        if not await self._send(node_id, frame):
            self._pending.pop(frame["id"], None)
            return {"ok": False, "error": "node not connected"}
        try:
            return await asyncio.wait_for(fut, timeout)
        except asyncio.TimeoutError:
            return {"ok": False, "error": "command timed out"}
        finally:
            self._pending.pop(frame["id"], None)

    # -- WebSocket handler ------------------------------------------------
    async def handle_ws(self, ws: WebSocket) -> None:
        if not self.check_key(self._key_from_ws(ws)):
            await ws.close(code=4401)
            return
        await ws.accept()
        node_id: Optional[str] = None
        try:
            # first frame must be register
            raw = await ws.receive_text()
            frame = protocol.decode(raw)
            if frame["type"] != protocol.REGISTER:
                await ws.send_text(protocol.encode(
                    protocol.error("expected_register", "first frame must be register")))
                await ws.close(code=4400)
                return
            node = self.registry.register(frame["body"])
            node_id = node.node_id
            self._conns[node_id] = ws
            log.info("node %s registered: roles=%s gpus=%d", node_id, node.roles, len(node.gpus))
            eff = self.config.effective(node_id, node.detected)
            await ws.send_text(protocol.encode(protocol.hello(
                eff, self._desired_state(node_id),
                heartbeat_sec=int(eff.get("ccp_heartbeat_sec", 5)),
                telemetry_sec=int(eff.get("ccp_telemetry_sec", 10)),
                reply_to=frame["id"],
            )))
            # main receive loop
            while True:
                raw = await ws.receive_text()
                try:
                    frame = protocol.decode(raw)
                except protocol.ProtocolError as exc:
                    await ws.send_text(protocol.encode(protocol.error("bad_frame", str(exc))))
                    continue
                self._dispatch(node_id, frame)
        except WebSocketDisconnect:
            pass
        except Exception as exc:  # noqa: BLE001
            log.debug("ws error for %s: %s", node_id, exc)
        finally:
            if node_id and self._conns.get(node_id) is ws:
                self._conns.pop(node_id, None)
                self.registry.mark_disconnected(node_id)
                log.info("node %s disconnected", node_id)

    def _dispatch(self, node_id: str, frame: dict) -> None:
        ftype = frame["type"]
        body = frame["body"]
        if ftype == protocol.HEARTBEAT:
            self.registry.on_heartbeat(node_id, body.get("seq", -1))
        elif ftype == protocol.TELEMETRY:
            self.registry.on_telemetry(node_id, body.get("gpus", []),
                                       body.get("replicas", []), body.get("mounts", []))
        elif ftype in (protocol.RESULT, protocol.ACK):
            rid = frame.get("reply_to")
            fut = self._pending.get(rid) if rid else None
            if ftype == protocol.RESULT and fut and not fut.done():
                fut.set_result(body)
        elif ftype == protocol.EVENT:
            log.info("event from %s: %s", node_id, body)
        elif ftype == protocol.REGISTER:
            self.registry.register(body)  # re-register (reconnect handled elsewhere)

    def _desired_state(self, node_id: str) -> dict:
        """Replicas this node should be running (for reconciliation)."""
        want = []
        for rid, meta in self._replicas.items():
            if node_id in meta.get("bodies", {}) and meta.get("state") not in ("STOPPED", "FAILED"):
                want.append(meta["bodies"][node_id])
        return {"replicas": want}

    # -- config push ------------------------------------------------------
    async def push_config(self, node_ids: Optional[list] = None) -> None:
        targets = node_ids or list(self._conns.keys())
        for nid in targets:
            node = self.registry.get(nid)
            if not node:
                continue
            eff = self.config.effective(nid, node.detected)
            await self._send(nid, protocol.make_frame(protocol.SET_CONFIG, {
                "effective_config": eff, "apply": "next_start",
            }))

    # -- launch / stop (Phase 2 scaffold) --------------------------------
    def plan(self, model: str, node_ids: list, *, port: int,
             num_layers: Optional[int] = None, **kw) -> dict:
        """Compute the replica plan WITHOUT executing it (dry run for the UI)."""
        nodes = [self.registry.get(n) for n in node_ids]
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return {"ok": False, "error": "no such participant nodes"}
        eff_by = {n.node_id: self.config.effective(n.node_id, n.detected) for n in nodes}
        try:
            plan = scheduler.build_replica_plan(
                kw.pop("replica_id", "plan-preview"), nodes, model, eff_by,
                port=port, num_layers=num_layers, **kw)
        except scheduler.ScheduleError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "layout": plan["layout"], "port": plan["port"],
                "pp_layer_partition": plan["pp_layer_partition"],
                "commands": [{"node_id": nid, "body": body} for nid, body in plan["commands"]]}

    async def _coordinate_storage(self, participants: list) -> dict:
        """Ensure a storage node is serving and each participant has the model mounted
        at the canonical path before a replica loads weights (docs/01, docs/03)."""
        storage = self.registry.storage_nodes(alive_only=True)
        if not storage:
            return {"ok": True, "detail": "no storage role; assuming models are local"}
        snode = storage[0]
        seff = self.config.effective(snode.node_id, snode.detected)
        canonical = seff.get("canonical_model_path", "/export/llm_models")
        export_dir = seff.get("storage_export_dir") or canonical
        fabric = (snode.addresses.get("fabric") or [snode.addresses.get("mgmt", "")])
        fabric_ip = fabric[0] if fabric else ""
        transport = seff.get("storage_transport", "tcp")
        results = {}
        if transport == "tcp":   # modelfsd; kernel-RDMA export is managed out of band
            listen = seff.get("storage_listen") or (fabric_ip + ":2049")
            results[snode.node_id] = await self.send_command(
                snode.node_id, protocol.make_frame(protocol.SERVE_STORAGE, {
                    "export_dir": export_dir, "listen": listen,
                    "allow": seff.get("storage_allow") or [],
                    "readahead": seff.get("storage_readahead")}), timeout=30)
        for p in participants:
            if p.node_id == snode.node_id:
                continue  # storage host reads locally, no mount
            peff = self.config.effective(p.node_id, p.detected)
            results[p.node_id] = await self.send_command(
                p.node_id, protocol.make_frame(protocol.MOUNT_STORAGE, {
                    "source": {"host": fabric_ip, "export": export_dir, "transport": transport},
                    "canonical_path": peff.get("canonical_model_path", canonical),
                    "opts": peff.get("mount_opts")}), timeout=45)
        ok = all(r.get("ok") for r in results.values())
        return {"ok": ok, "storage_host": snode.node_id, "transport": transport,
                "results": results}

    async def launch(self, model: str, node_ids: list, *, port: int,
                     num_layers: Optional[int] = None, **kw) -> dict:
        nodes = [self.registry.get(n) for n in node_ids]
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return {"ok": False, "error": "no such participant nodes"}
        if not kw.pop("skip_storage", False):
            storage_res = await self._coordinate_storage(nodes)
            if not storage_res.get("ok"):
                return {"ok": False, "error": "storage coordination failed",
                        "storage": storage_res}
        eff_by = {n.node_id: self.config.effective(n.node_id, n.detected) for n in nodes}
        replica_id = kw.pop("replica_id", None) or f"replica-{len(self._replicas) + 1}"
        try:
            plan = scheduler.build_replica_plan(
                replica_id, nodes, model, eff_by, port=port, num_layers=num_layers, **kw)
        except scheduler.ScheduleError as exc:
            return {"ok": False, "error": str(exc)}
        bodies = {nid: body for nid, body in plan["commands"]}
        self._replicas[replica_id] = {
            "state": "STARTING", "node_ids": [n.node_id for n in nodes],
            "bodies": bodies, "layout": plan["layout"], "port": port, "model": model,
        }
        # head last so workers are ready first
        order = [nid for nid in plan["layout"]["ordered_ids"]][::-1]
        results = {}
        for nid in order:
            frame = protocol.make_frame(protocol.ENSURE_REPLICA, bodies[nid])
            results[nid] = await self.send_command(nid, frame, timeout=kw.get("timeout", 300))
        ok = all(r.get("ok") for r in results.values())
        self._replicas[replica_id]["state"] = "SERVING" if ok else "FAILED"
        return {"ok": ok, "replica_id": replica_id, "layout": plan["layout"], "results": results}

    async def stop_replica(self, replica_id: str, ray: bool = True) -> dict:
        meta = self._replicas.get(replica_id)
        if not meta:
            return {"ok": False, "error": "no such replica"}
        results = {}
        for nid in meta.get("node_ids", []):
            frame = protocol.make_frame(protocol.STOP_REPLICA,
                                        {"replica_id": replica_id, "ray": ray})
            results[nid] = await self.send_command(nid, frame, timeout=60)
        meta["state"] = "STOPPED"
        return {"ok": True, "replica_id": replica_id, "results": results}

    def summary(self) -> dict:
        nodes = self.registry.all()
        alive = [n for n in nodes if n.state not in (protocol.STATE_DOWN,
                                                     protocol.STATE_DISCONNECTED)]
        gpus = sum(len(n.gpus) for n in alive)
        return {
            "nodes_total": len(nodes),
            "nodes_alive": len(alive),
            "gpus_total": gpus,
            "roles": {r: len([n for n in alive if r in n.roles]) for r in protocol.ALL_ROLES},
            "replicas": {rid: m.get("state") for rid, m in self._replicas.items()},
        }


# ---------------------------------------------------------------------------
# Router factory — mounted by admin/app.py when CONTROL_PLANE is enabled.
# ---------------------------------------------------------------------------

def build_cluster_router(hub: ClusterHub) -> APIRouter:
    router = APIRouter()

    def _auth(x_api_key: Optional[str], authorization: Optional[str]) -> bool:
        key = x_api_key
        if not key and authorization and authorization.lower().startswith("bearer "):
            key = authorization[7:].strip()
        return hub.check_key(key)

    @router.websocket("/api/ccp")
    async def ccp(ws: WebSocket):
        await hub.handle_ws(ws)

    @router.get("/api/cluster/nodes")
    async def nodes(x_api_key: Optional[str] = Header(None),
                    authorization: Optional[str] = Header(None)):
        if not _auth(x_api_key, authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return hub.registry.public()

    @router.get("/api/cluster/summary")
    async def summary(x_api_key: Optional[str] = Header(None),
                      authorization: Optional[str] = Header(None)):
        if not _auth(x_api_key, authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return hub.summary()

    @router.get("/api/cluster/config")
    async def get_config(x_api_key: Optional[str] = Header(None),
                         authorization: Optional[str] = Header(None)):
        if not _auth(x_api_key, authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        return hub.config.snapshot(hub.registry.detected_by_node())

    @router.post("/api/cluster/config")
    async def set_config(request: Request, x_api_key: Optional[str] = Header(None),
                         authorization: Optional[str] = Header(None)):
        if not _auth(x_api_key, authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        data = await request.json()
        errors = []
        if "defaults" in data:
            errors += hub.config.set_defaults(data["defaults"] or {})
        for nid, nb in (data.get("nodes") or {}).items():
            node = hub.registry.get(nid)
            det = node.detected if node else None
            errors += hub.config.set_override(nid, (nb or {}).get("overrides", {}), det)
        if errors:
            return JSONResponse(status_code=400, content={"error": "; ".join(errors)})
        await hub.push_config(list((data.get("nodes") or {}).keys()) or None)
        return hub.config.snapshot(hub.registry.detected_by_node())

    @router.post("/api/cluster/plan")
    async def plan(request: Request, x_api_key: Optional[str] = Header(None),
                   authorization: Optional[str] = Header(None)):
        if not _auth(x_api_key, authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        d = await request.json()
        res = hub.plan(
            d["model"], d["node_ids"], port=int(d.get("port", 8001)),
            num_layers=d.get("num_layers"), served_model_name=d.get("served_model_name"),
            max_model_len=d.get("max_model_len"), extra_args=d.get("extra_args"))
        return JSONResponse(status_code=200 if res.get("ok") else 400, content=res)

    @router.post("/api/cluster/launch")
    async def launch(request: Request, x_api_key: Optional[str] = Header(None),
                     authorization: Optional[str] = Header(None)):
        if not _auth(x_api_key, authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        d = await request.json()
        res = await hub.launch(
            d["model"], d["node_ids"], port=int(d["port"]),
            num_layers=d.get("num_layers"), served_model_name=d.get("served_model_name"),
            max_model_len=d.get("max_model_len"), extra_args=d.get("extra_args"))
        return JSONResponse(status_code=200 if res.get("ok") else 400, content=res)

    @router.post("/api/cluster/stop")
    async def stop(request: Request, x_api_key: Optional[str] = Header(None),
                   authorization: Optional[str] = Header(None)):
        if not _auth(x_api_key, authorization):
            return JSONResponse(status_code=401, content={"error": "unauthorized"})
        d = await request.json()
        return await hub.stop_replica(d["replica_id"], ray=d.get("ray", True))

    return router
