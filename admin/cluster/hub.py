"""CCP hub — the admin's control-plane server.

Provides a FastAPI router with:
  * ``WS  /api/ccp``            the Cluster Control Protocol endpoint (agents dial in here)
  * ``GET /api/cluster/nodes``  live node + GPU inventory (for the UI monitor)
  * ``GET /api/cluster/config`` layered config snapshot (defaults + per-node)
  * ``POST /api/cluster/config`` update defaults / per-node overrides, pushed to agents
  * ``GET /api/cluster/summary`` quick counts
  * ``POST /api/cluster/launch`` schedule a distributed instance across participants (Phase 2)
  * ``POST /api/cluster/stop``   stop an instance

The hub owns a Registry and a ClusterConfigStore, tracks one live WebSocket per node, and runs a
background sweep that promotes silent nodes to DOWN.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
import time
from typing import Optional

from fastapi import APIRouter, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

from . import protocol, scheduler
from .config import ClusterConfigStore
from .registry import Registry

log = logging.getLogger("admin.cluster.hub")


class ClusterHub:
    def __init__(self, config_path: str, api_key: str = "", auth_enabled: bool = True,
                 agent_version: str = "0.4.0", cookie_name: str = "vllm_admin_session"):
        # Browser sessions authenticate with this cookie (set by /api/auth/login), the
        # same credential the app's auth middleware accepts. The router must honour it
        # or every UI data call 401s even though the page itself loaded.
        self.cookie_name = cookie_name
        self.registry = Registry()
        self.config = ClusterConfigStore(config_path)
        self.api_key = api_key
        self.auth_enabled = auth_enabled
        self.agent_version = agent_version
        self._conns: dict = {}                 # node_id -> WebSocket
        self._pending: dict = {}               # frame id -> asyncio.Future (result)
        self._instances: dict = {}              # instance_id -> plan metadata
        self._sweep_task: Optional[asyncio.Task] = None
        self._stop = False

    # -- join tokens (node onboarding) ------------------------------------
    @property
    def _tokens_path(self) -> str:
        return os.path.join(os.path.dirname(self.config.path) or ".", "join-tokens.json")

    def _load_tokens(self) -> dict:
        try:
            with open(self._tokens_path, "r", encoding="utf-8") as fh:
                return json.load(fh) or {}
        except (FileNotFoundError, ValueError, OSError):
            return {}

    def _save_tokens(self, toks: dict) -> None:
        try:
            os.makedirs(os.path.dirname(self._tokens_path) or ".", exist_ok=True)
            with open(self._tokens_path, "w", encoding="utf-8") as fh:
                json.dump(toks, fh, indent=2)
        except OSError as exc:
            log.warning("could not persist join tokens: %s", exc)

    def mint_join_token(self, roles: list, ttl_minutes: int = 60,
                        label: str = "") -> dict:
        """A short-lived credential that lets ONE new node join. It is deliberately
        not the admin API key — you never paste that onto a new box."""
        roles = protocol.validate_roles(roles or [protocol.ROLE_PARTICIPANT])
        token = "join-" + secrets.token_urlsafe(24)
        toks = self._load_tokens()
        toks[token] = {"roles": roles, "label": label, "created": time.time(),
                       "expires": time.time() + ttl_minutes * 60, "node_id": None}
        self._save_tokens(toks)
        return {"token": token, "roles": roles, "expires_in_min": ttl_minutes}

    def _join_token_record(self, key: Optional[str]) -> Optional[dict]:
        if not key or not key.startswith("join-"):
            return None
        rec = self._load_tokens().get(key)
        if not rec:
            return None
        # Unused tokens expire; once bound to a node it stays valid for reconnects
        # until explicitly revoked.
        if rec.get("node_id") is None and time.time() > rec.get("expires", 0):
            return None
        return rec

    def bind_join_token(self, key: str, node_id: str) -> None:
        toks = self._load_tokens()
        rec = toks.get(key)
        if rec and rec.get("node_id") in (None, node_id):
            rec["node_id"] = node_id
            self._save_tokens(toks)

    def revoke_join_token(self, token: Optional[str] = None,
                          node_id: Optional[str] = None) -> dict:
        toks = self._load_tokens()
        gone = [k for k, v in toks.items()
                if (token and k == token) or (node_id and v.get("node_id") == node_id)]
        for k in gone:
            toks.pop(k, None)
        self._save_tokens(toks)
        return {"ok": bool(gone), "revoked": len(gone)}

    def list_join_tokens(self) -> list:
        out = []
        for k, v in self._load_tokens().items():
            out.append({"token": k[:14] + "…", "roles": v.get("roles"),
                        "label": v.get("label", ""), "node_id": v.get("node_id"),
                        "expires": v.get("expires"),
                        "used": v.get("node_id") is not None})
        return out

    # -- auth -------------------------------------------------------------
    def check_key(self, presented: Optional[str]) -> bool:
        if not self.auth_enabled:
            return True
        if presented and presented == self.api_key:
            return True
        # a valid join token authenticates the CCP connection of a joining node
        return self._join_token_record(presented) is not None

    def _key_from_ws(self, ws: WebSocket) -> Optional[str]:
        auth = ws.headers.get("authorization", "")
        if auth.lower().startswith("bearer "):
            return auth[7:].strip()
        return (ws.headers.get("x-api-key")
                or ws.query_params.get("key")
                or ws.cookies.get(self.cookie_name))

    def key_from_request(self, request: Request) -> Optional[str]:
        """Credentials accepted for the cluster REST API, in the same order the app's
        auth middleware uses: X-API-Key header, Bearer token, then session cookie."""
        key = request.headers.get("x-api-key")
        if not key:
            auth = request.headers.get("authorization", "")
            if auth.lower().startswith("bearer "):
                key = auth[7:].strip()
        if not key:
            key = request.cookies.get(self.cookie_name)
        return key.strip() if key else None

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
        # mark any instance that depends on this node as failed (hold+alert policy, docs/06)
        for rid, meta in self._instances.items():
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
        presented = self._key_from_ws(ws)
        if not self.check_key(presented):
            await ws.close(code=4401)
            return
        join_rec = self._join_token_record(presented)   # None when using the admin key
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
            if join_rec is not None:
                # A node onboarded with a join token takes the roles the token was
                # minted for, and the token binds to it for future reconnects.
                self.bind_join_token(presented, node_id)
                override = self.config.override(node_id)
                if not override.get("roles"):
                    override["roles"] = join_rec.get("roles") or [protocol.ROLE_PARTICIPANT]
                    self.config.set_override(node_id, override, node.detected)
                log.info("node %s joined via token (roles=%s)", node_id, override.get("roles"))
            self._apply_role_override(node)   # admin-assigned roles win over the agent's .env
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
                                       body.get("instances", []), body.get("mounts", []),
                                       volumes=body.get("volumes"),
                                       shares=body.get("shares"))
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
        """Instances this node should be running (for reconciliation)."""
        want = []
        for rid, meta in self._instances.items():
            if node_id in meta.get("bodies", {}) and meta.get("state") not in ("STOPPED", "FAILED"):
                want.append(meta["bodies"][node_id])
        return {"instances": want}

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
    def _resolve_model(self, model: str, nodes: list) -> str:
        """Turn a bare model NAME into a path under the canonical model dir.

        /api/models returns names ("Devstral-Small-2507"), but vLLM's --model wants a
        filesystem path or a HF repo id. v0.3.0 resolved this in /api/start; the cluster
        path must do the same or the launch dies with "not a local folder".
        Anything already absolute, or that looks like a HF repo id (org/name), is left
        alone.
        """
        if not model or model.startswith("/") or "/" in model.strip("/"):
            return model
        base = ""
        for n in nodes:
            base = n.canonical_model_path or base
            if base:
                break
        base = base or self.config.defaults().get("canonical_model_path") or "/models"
        return base.rstrip("/") + "/" + model

    def plan(self, model: str, node_ids: list, *, port: int,
             num_layers: Optional[int] = None, **kw) -> dict:
        """Compute the instance plan WITHOUT executing it (dry run for the UI)."""
        nodes = [self.registry.get(n) for n in node_ids]
        nodes = [n for n in nodes if n is not None]
        if not nodes:
            return {"ok": False, "error": "no such participant nodes"}
        eff_by = {n.node_id: self.config.effective(n.node_id, n.detected) for n in nodes}
        model = self._resolve_model(model, nodes)
        try:
            plan = scheduler.build_instance_plan(
                kw.pop("instance_id", "plan-preview"), nodes, model, eff_by,
                port=port, num_layers=num_layers, **kw)
        except scheduler.ScheduleError as exc:
            return {"ok": False, "error": str(exc)}
        return {"ok": True, "layout": plan["layout"], "port": plan["port"],
                "pp_layer_partition": plan["pp_layer_partition"],
                "commands": [{"node_id": nid, "body": body} for nid, body in plan["commands"]]}

    async def _coordinate_storage(self, participants: list) -> dict:
        """Ensure a storage node is serving and each participant has the model mounted
        at the canonical path before an instance loads weights (docs/01, docs/03)."""
        storage = self.registry.storage_nodes(alive_only=True)
        if not storage:
            return {"ok": True, "detail": "no storage role; assuming models are local"}
        snode = storage[0]
        seff = self.config.effective(snode.node_id, snode.detected)
        # The agent knows its own in-container path (reported at register); trust that
        # over any configured default, which may be a host path the agent cannot see.
        canonical = (snode.canonical_model_path
                     or seff.get("canonical_model_path", "/models"))
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
                    "canonical_path": (p.canonical_model_path
                                       or peff.get("canonical_model_path", canonical)),
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
        model = self._resolve_model(model, nodes)
        instance_id = kw.pop("instance_id", None) or f"instance-{len(self._instances) + 1}"
        try:
            plan = scheduler.build_instance_plan(
                instance_id, nodes, model, eff_by, port=port, num_layers=num_layers, **kw)
        except scheduler.ScheduleError as exc:
            return {"ok": False, "error": str(exc)}
        bodies = {nid: body for nid, body in plan["commands"]}
        self._instances[instance_id] = {
            "state": "STARTING", "node_ids": [n.node_id for n in nodes],
            "bodies": bodies, "layout": plan["layout"], "port": port, "model": model,
        }
        # head last so workers are ready first
        order = [nid for nid in plan["layout"]["ordered_ids"]][::-1]
        results = {}
        for nid in order:
            frame = protocol.make_frame(protocol.ENSURE_INSTANCE, bodies[nid])
            results[nid] = await self.send_command(nid, frame, timeout=kw.get("timeout", 300))
        ok = all(r.get("ok") for r in results.values())
        self._instances[instance_id]["state"] = "SERVING" if ok else "FAILED"
        return {"ok": ok, "instance_id": instance_id, "layout": plan["layout"], "results": results}

    async def stop_instance(self, instance_id: str, ray: bool = True) -> dict:
        meta = self._instances.get(instance_id)
        if not meta:
            return {"ok": False, "error": "no such instance"}
        results = {}
        for nid in meta.get("node_ids", []):
            frame = protocol.make_frame(protocol.STOP_INSTANCE,
                                        {"instance_id": instance_id, "ray": ray})
            results[nid] = await self.send_command(nid, frame, timeout=60)
        meta["state"] = "STOPPED"
        return {"ok": True, "instance_id": instance_id, "results": results}

    # -- logs -------------------------------------------------------------
    async def instance_logs(self, instance_id: str, tail: int = 200,
                            node_id: Optional[str] = None) -> dict:
        """Fetch an instance's logs from every node running it, merged and
        node-tagged — the cluster equivalent of tailing one process. Without this
        a failed launch can only be diagnosed by shelling into a container."""
        meta = self._instances.get(instance_id) or {}
        targets = [node_id] if node_id else list(meta.get("node_ids") or [])
        if not targets:   # unknown instance: ask every connected node
            targets = list(self._conns.keys())
        out, merged = {}, []
        for nid in targets:
            res = await self.send_command(nid, protocol.make_frame(
                protocol.GET_LOGS, {"instance_id": instance_id, "tail": tail}), timeout=20)
            out[nid] = res
            for line in (res.get("lines") or []):
                merged.append({"node_id": nid, "line": line})
        return {"ok": True, "instance_id": instance_id,
                "nodes": out, "merged": merged,
                "state": meta.get("state"), "model": meta.get("model")}

    # -- role management --------------------------------------------------
    async def set_roles(self, node_id: str, roles: list) -> dict:
        """Reassign a node's roles from the admin. The admin-assigned set wins over the
        roles the agent reported from its own .env, and is re-applied on reconnect."""
        node = self.registry.get(node_id)
        if not node:
            return {"ok": False, "error": "no such node"}
        try:
            roles = protocol.validate_roles(roles)
        except protocol.ProtocolError as exc:
            return {"ok": False, "error": str(exc)}

        # Guards — refuse changes that would break the cluster.
        if protocol.ROLE_ADMIN in node.roles and protocol.ROLE_ADMIN not in roles:
            others = [n for n in self.registry.all()
                      if n.node_id != node_id and protocol.ROLE_ADMIN in n.roles]
            if not others:
                return {"ok": False, "error": "refusing to remove the only admin role"}
        if protocol.ROLE_PARTICIPANT in node.roles and protocol.ROLE_PARTICIPANT not in roles:
            busy = [r for r in (node.instances or []) if r.get("state") == "SERVING"]
            if busy:
                return {"ok": False,
                        "error": "node is serving %d instance(s); stop them first" % len(busy)}
        # Losing storage: stop any exports first so we don't strand mounts.
        dropped_storage = (protocol.ROLE_STORAGE in node.roles
                           and protocol.ROLE_STORAGE not in roles)
        if dropped_storage:
            for sh in list(node.shares or []):
                await self.send_command(node_id, protocol.make_frame(
                    protocol.UNSHARE_STORAGE, {"export_dir": sh.get("path")}), timeout=30)

        override = self.config.override(node_id)
        override["roles"] = roles
        errs = self.config.set_override(node_id, override, node.detected)
        if errs:
            return {"ok": False, "error": "; ".join(errs)}
        node.roles = roles                      # take effect immediately
        await self.push_config([node_id])       # and tell the agent
        return {"ok": True, "roles": roles}

    def _apply_role_override(self, node) -> None:
        """Admin-assigned roles win over what the agent reported (applied on register)."""
        eff = self.config.effective(node.node_id, node.detected)
        assigned = eff.get("roles")
        if assigned:
            try:
                node.roles = protocol.validate_roles(assigned)
            except protocol.ProtocolError:
                pass

    # -- storage shares ---------------------------------------------------
    def _fabric_ip(self, node) -> str:
        fabric = (node.addresses or {}).get("fabric") or []
        return fabric[0] if fabric else (node.addresses or {}).get("mgmt", "")

    def _cluster_client_cidrs(self) -> list:
        """Default allow-list for a share: every known cluster member address as /32.
        modelfsd requires an explicit --allow, and this keeps it tight (cluster only)
        without the user having to configure subnets by hand."""
        cidrs = []
        for n in self.registry.all():
            addrs = (n.addresses or {})
            for ip in ([addrs.get("mgmt")] + list(addrs.get("fabric") or [])):
                if ip and "/" not in ip:
                    cidr = ip + "/32"
                    if cidr not in cidrs:
                        cidrs.append(cidr)
        return cidrs

    def _mounted_by(self, export_path: str, source_node_id: str) -> list:
        out = []
        for n in self.registry.all():
            if n.node_id == source_node_id:
                continue
            for m in (n.mounts or []):
                src = m.get("source") or ""
                if src.endswith(":" + export_path) or src == export_path:
                    out.append({"node_id": n.node_id, "path": m.get("path"),
                                "ok": bool(m.get("ok"))})
        return out

    def shares_catalog(self) -> dict:
        """{shares:[...], volumes_by_node:{...}} — drives the Storage/Workers tabs."""
        volumes_by_node, shares = {}, []
        for n in self.registry.all():
            live = {s.get("path"): s for s in (n.shares or [])}
            wanted = set(self.config.effective(n.node_id, n.detected).get("storage_shares") or [])
            vol_paths = {v.get("path") for v in (n.volumes or [])}
            vols = []
            for v in (n.volumes or []):
                p = v.get("path")
                sh = live.get(p)
                item = dict(v)
                item["shared"] = bool(sh) or p in wanted
                item["endpoint"] = (sh or {}).get("endpoint")
                vols.append(item)
            # user-added paths that aren't whole volumes
            for p in sorted(wanted | set(live)):
                if p in vol_paths:
                    continue
                sh = live.get(p)
                vols.append({"path": p, "fstype": "", "total": 0, "used": 0, "free": 0,
                             "shared": True, "custom": True,
                             "endpoint": (sh or {}).get("endpoint")})
            volumes_by_node[n.node_id] = vols
            for p, sh in live.items():
                base = next((v for v in (n.volumes or []) if v.get("path") == p), {})
                shares.append({
                    "id": "%s:%s" % (n.node_id, p),
                    "node_id": n.node_id, "path": p,
                    "endpoint": sh.get("endpoint") or self._fabric_ip(n),
                    "fstype": base.get("fstype", ""), "total": base.get("total", 0),
                    "used": base.get("used", 0), "free": base.get("free", 0),
                    "ok": bool(sh.get("ok", True)),
                    "mounted_by": self._mounted_by(p, n.node_id),
                })
        return {"shares": shares, "volumes_by_node": volumes_by_node}

    async def set_share(self, node_id: str, path: str, enabled: bool) -> dict:
        node = self.registry.get(node_id)
        if not node:
            return {"ok": False, "error": "no such node"}
        if protocol.ROLE_STORAGE not in node.roles:
            return {"ok": False, "error": "node does not have the storage role"}
        eff = self.config.effective(node_id, node.detected)
        override = self.config.override(node_id)
        current = list(override.get("storage_shares") or eff.get("storage_shares") or [])
        if enabled and path not in current:
            current.append(path)
        elif not enabled and path in current:
            current.remove(path)
        if enabled:
            frame = protocol.make_frame(protocol.SERVE_STORAGE, {
                # user-selected NIC wins; else default to the RDMA fabric IP
                "export_dir": path,
                "listen": eff.get("storage_bind_ip") or self._fabric_ip(node),
                # explicit config wins; otherwise allow exactly the cluster's members
                "allow": eff.get("storage_allow") or self._cluster_client_cidrs(),
                "readahead": eff.get("storage_readahead"),
            })
        else:
            frame = protocol.make_frame(protocol.UNSHARE_STORAGE, {"export_dir": path})
        res = await self.send_command(node_id, frame, timeout=30)
        if not res.get("ok"):
            return {"ok": False, "error": res.get("error") or res.get("detail") or "failed"}
        override["storage_shares"] = current
        errs = self.config.set_override(node_id, override, node.detected)
        if errs:
            return {"ok": False, "error": "; ".join(errs)}
        return {"ok": True, "detail": res.get("detail", ""), "shares": current}

    async def set_mount(self, node_id: str, share_id: str, enabled: bool,
                        mount_path: Optional[str] = None) -> dict:
        node = self.registry.get(node_id)
        if not node:
            return {"ok": False, "error": "no such node"}
        cat = self.shares_catalog()
        share = next((s for s in cat["shares"] if s["id"] == share_id), None)
        if not share:
            return {"ok": False, "error": "no such share"}
        target = mount_path or share["path"]
        if enabled:
            src_node = self.registry.get(share["node_id"])
            host = share["endpoint"].rsplit(":", 1)[0] if ":" in share["endpoint"] else share["endpoint"]
            peff = self.config.effective(node_id, node.detected)
            frame = protocol.make_frame(protocol.MOUNT_STORAGE, {
                "source": {"host": host, "export": share["path"],
                           "transport": peff.get("storage_transport", "tcp"),
                           "port": share["endpoint"].rsplit(":", 1)[-1]},
                "canonical_path": target, "opts": peff.get("mount_opts")})
        else:
            frame = protocol.make_frame(protocol.UNMOUNT_STORAGE, {"canonical_path": target})
        res = await self.send_command(node_id, frame, timeout=60)
        return {"ok": bool(res.get("ok")),
                "error": res.get("error", ""), "detail": res.get("detail", "")}

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
            # display names for the UI; wire ids stay stable
            "role_labels": protocol.ROLE_LABELS,
            "instances": {rid: m.get("state") for rid, m in self._instances.items()},
        }


# ---------------------------------------------------------------------------
# Router factory — mounted by admin/app.py when CONTROL_PLANE is enabled.
# ---------------------------------------------------------------------------

def build_cluster_router(hub: ClusterHub) -> APIRouter:
    router = APIRouter()

    def _auth(request: Request) -> bool:
        return hub.check_key(hub.key_from_request(request))

    _UNAUTH = JSONResponse(status_code=401, content={"error": "unauthorized"})

    @router.websocket("/api/ccp")
    async def ccp(ws: WebSocket):
        await hub.handle_ws(ws)

    @router.get("/api/cluster/nodes")
    async def nodes(request: Request):
        if not _auth(request):
            return _UNAUTH
        return hub.registry.public()

    @router.get("/api/cluster/summary")
    async def summary(request: Request):
        if not _auth(request):
            return _UNAUTH
        return hub.summary()

    @router.get("/api/cluster/config")
    async def get_config(request: Request):
        if not _auth(request):
            return _UNAUTH
        return hub.config.snapshot(hub.registry.detected_by_node())

    @router.post("/api/cluster/config")
    async def set_config(request: Request):
        if not _auth(request):
            return _UNAUTH
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

    @router.get("/api/cluster/image")
    async def image_ref(request: Request):
        # join.sh calls this with its join token to learn which image to pull, so the
        # whole cluster runs a byte-identical build.
        if not _auth(request):
            return _UNAUTH
        return {"image_ref": os.getenv("IMAGE_REF", "")}

    @router.get("/api/cluster/join-tokens")
    async def join_tokens(request: Request):
        if not _auth(request):
            return _UNAUTH
        return hub.list_join_tokens()

    @router.post("/api/cluster/join-token")
    async def join_token(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        try:
            tok = hub.mint_join_token(d.get("roles") or ["participant"],
                                      int(d.get("ttl_minutes", 60)), d.get("label", ""))
        except protocol.ProtocolError as exc:
            return JSONResponse(status_code=400, content={"ok": False, "error": str(exc)})
        # The one-liner to paste on the new box. The image ref is baked in rather than
        # fetched back from the admin: the join token is not accepted by the app's auth
        # middleware (only the CCP WebSocket honours it), and this saves a round trip.
        base = str(request.base_url).rstrip("/")
        cmd = ("curl -sfL %s/join.sh | sudo bash -s -- --token %s --admin-url %s"
               % (base, tok["token"], base))
        roles_arg = ",".join(tok["roles"])
        if roles_arg and roles_arg != "participant":
            cmd += " --roles %s" % roles_arg
        image_ref = os.getenv("IMAGE_REF", "")
        if image_ref:
            cmd += " --image %s" % image_ref
        tok["image_ref"] = image_ref
        tok["join_command"] = cmd
        tok["ok"] = True
        return tok

    @router.post("/api/cluster/join-token/revoke")
    async def revoke_join_token(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        return hub.revoke_join_token(d.get("token"), d.get("node_id"))

    @router.get("/api/cluster/logs")
    async def logs(request: Request):
        if not _auth(request):
            return _UNAUTH
        q = request.query_params
        iid = q.get("instance") or q.get("instance_id") or ""
        if not iid:
            return JSONResponse(status_code=400,
                                content={"ok": False, "error": "instance= is required"})
        try:
            tail = int(q.get("tail", 200))
        except ValueError:
            tail = 200
        return await hub.instance_logs(iid, tail, q.get("node"))

    @router.post("/api/cluster/roles")
    async def roles(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        res = await hub.set_roles(d["node_id"], d.get("roles") or [])
        return JSONResponse(status_code=200 if res.get("ok") else 400, content=res)

    @router.get("/api/cluster/shares")
    async def shares(request: Request):
        if not _auth(request):
            return _UNAUTH
        return hub.shares_catalog()

    @router.post("/api/cluster/share")
    async def share(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        res = await hub.set_share(d["node_id"], d["path"], bool(d.get("enabled", True)))
        return JSONResponse(status_code=200 if res.get("ok") else 400, content=res)

    @router.post("/api/cluster/mount")
    async def mount(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        res = await hub.set_mount(d["node_id"], d["share_id"], bool(d.get("enabled", True)),
                                  d.get("mount_path"))
        return JSONResponse(status_code=200 if res.get("ok") else 400, content=res)

    @router.post("/api/cluster/plan")
    async def plan(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        res = hub.plan(
            d["model"], d["node_ids"], port=int(d.get("port", 8001)),
            num_layers=d.get("num_layers"), served_model_name=d.get("served_model_name"),
            max_model_len=d.get("max_model_len"), extra_args=d.get("extra_args"))
        return JSONResponse(status_code=200 if res.get("ok") else 400, content=res)

    @router.post("/api/cluster/launch")
    async def launch(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        res = await hub.launch(
            d["model"], d["node_ids"], port=int(d["port"]),
            num_layers=d.get("num_layers"), served_model_name=d.get("served_model_name"),
            max_model_len=d.get("max_model_len"), extra_args=d.get("extra_args"))
        return JSONResponse(status_code=200 if res.get("ok") else 400, content=res)

    @router.post("/api/cluster/stop")
    async def stop(request: Request):
        if not _auth(request):
            return _UNAUTH
        d = await request.json()
        return await hub.stop_instance(d["instance_id"], ray=d.get("ray", True))

    return router
