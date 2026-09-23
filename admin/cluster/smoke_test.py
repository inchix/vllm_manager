"""End-to-end smoke test for the v0.4.0 control plane.

Starts a minimal admin (CCP hub + REST) on a real port, runs the real node agent against it,
and asserts the node registers and streams live GPU telemetry. Runnable standalone::

    python3 admin/cluster/smoke_test.py

Requires: fastapi, uvicorn, websockets (all in the vLLM image / present on this dev host).
Uses the local machine's real GPUs for the telemetry check when available; still passes on a
GPU-less box (telemetry list is just empty).
"""
from __future__ import annotations

import asyncio
import json
import os
import socket
import sys
import tempfile
import threading
import time
import urllib.request

# make `admin` importable when run from the repo root
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fastapi import FastAPI  # noqa: E402
import uvicorn  # noqa: E402

from admin.cluster.hub import ClusterHub, build_cluster_router  # noqa: E402

KEY = "smoke-test-key"


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def _get(url: str) -> object:
    req = urllib.request.Request(url, headers={"X-API-Key": KEY})
    with urllib.request.urlopen(req, timeout=5) as r:
        return json.loads(r.read().decode())


def _make_app(cfg_path: str):
    from contextlib import asynccontextmanager

    hub = ClusterHub(config_path=cfg_path, api_key=KEY, auth_enabled=True)

    @asynccontextmanager
    async def lifespan(_app):
        hub.start()
        try:
            yield
        finally:
            await hub.stop()

    app = FastAPI(lifespan=lifespan)
    app.include_router(build_cluster_router(hub))
    return app, hub


def _serve(app, port: int):
    cfg = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(cfg)
    server.run()


async def _run_agent(port: int, seconds: float):
    os.environ["CCP_ADMIN_URL"] = f"ws://127.0.0.1:{port}"
    os.environ["ADMIN_API_KEY"] = KEY
    os.environ["ROLES"] = "participant,storage"
    os.environ["NODE_ID"] = "smoke-node"
    os.environ["CLUSTER_ID"] = "default"
    # import after env is set
    from admin.agent.agentd import Agent
    agent = Agent()
    task = asyncio.ensure_future(agent.run())
    await asyncio.sleep(seconds)          # connect + stream telemetry
    # snapshot the cluster WHILE the agent is still connected
    loop = asyncio.get_event_loop()
    base = f"http://127.0.0.1:{port}"
    live = {
        "nodes": await loop.run_in_executor(None, _get, base + "/api/cluster/nodes"),
        "summary": await loop.run_in_executor(None, _get, base + "/api/cluster/summary"),
        "config": await loop.run_in_executor(None, _get, base + "/api/cluster/config"),
    }
    agent._stop = True
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass
    return live


def main() -> int:
    tmp = tempfile.mkdtemp()
    cfg_path = os.path.join(tmp, "cluster-config.json")
    port = _free_port()
    app, hub = _make_app(cfg_path)

    server_thread = threading.Thread(target=_serve, args=(app, port), daemon=True)
    server_thread.start()

    # wait for the server
    base = f"http://127.0.0.1:{port}"
    for _ in range(50):
        try:
            _get(base + "/api/cluster/summary")
            break
        except Exception:
            time.sleep(0.1)
    else:
        print("FAIL: server did not come up")
        return 1

    # run the real agent against it; capture a snapshot while connected
    live = asyncio.run(_run_agent(port, seconds=8.0))
    nodes = live["nodes"]
    summary = live["summary"]
    cfg = live["config"]

    ok = True
    print("\n=== summary ===")
    print(json.dumps(summary, indent=2))
    print("=== nodes ===")
    print(json.dumps(nodes, indent=2)[:1600])

    def check(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + ": " + name)
        ok = ok and cond

    node = next((n for n in nodes if n["node_id"] == "smoke-node"), None)
    check("node registered", node is not None)
    if node:
        check("roles include participant+storage",
              set(node["roles"]) >= {"participant", "storage"})
        check("state is READY/SERVING", node["state"] in ("READY", "SERVING"))
        check("addresses reported", bool(node.get("addresses")))
        # telemetry: GPUs present only on a GPU box; assert the pipe works either way
        gpus = node.get("gpus", [])
        if gpus:
            live = any(g.get("util") is not None or g.get("mem_used") is not None for g in gpus)
            check("live GPU telemetry flowing", live)
        else:
            print("NOTE: no GPUs on this host; telemetry pipe exercised with empty list")
    check("summary counts a node", summary["nodes_total"] >= 1)
    check("config snapshot has this node's detected inventory",
          "smoke-node" in cfg.get("nodes", {}))

    print("\n" + ("ALL PASS" if ok else "SOME CHECKS FAILED"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
