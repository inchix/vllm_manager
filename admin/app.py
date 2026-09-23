import asyncio
import json as json_mod
import logging
import os
import shlex
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel

from admin import auth, persistence
from admin.vllm_manager import State, VllmConfig, VllmManager

logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------- Constants ----------------

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))

# Multi-node: set by run.sh when CLUSTER_MODE=true. Enables the /api/cluster
# endpoint and the "use remote GPUs" launch path (Ray schedules workers across
# boxes). entrypoint.sh has already started a Ray head on this container.
CLUSTER_MODE = os.getenv("CLUSTER_MODE", "false").lower() == "true"

# v0.4.0 control plane (agent/CCP + composable roles). Opt-in; default off keeps
# the exact v0.3.0 behaviour. See docs/README.md.
CONTROL_PLANE = os.getenv("CONTROL_PLANE", "false").lower() == "true"
cluster_hub = None  # assigned below when CONTROL_PLANE is enabled

VLLM_PORT_START = int(os.getenv("ADMIN_VLLM_PORT_START", os.getenv("VLLM_PORT_START", "8001")))
VLLM_PORT_END = int(os.getenv("ADMIN_VLLM_PORT_END", os.getenv("VLLM_PORT_END", "8010")))
MAX_INSTANCES = VLLM_PORT_END - VLLM_PORT_START + 1

ALLOWED_DTYPES = {"auto", "float16", "bfloat16", "float32"}
ALLOWED_MODEL_IMPLS = {"auto", "transformers", "vllm"}
ALLOWED_TOOL_PARSERS = {
    "hermes", "llama3_json", "llama4_json", "mistral",
    "qwen3_xml", "qwen3_coder", "deepseek_v3", "pythonic", "openai",
}

# Flags we allow callers to pass via `extra_args`. Anything here is assumed
# safe to expose; flags that change security posture (remote code exec,
# load-format, executor backend, host/port rebinding, etc.) are deliberately
# omitted.
_EXTRA_ARGS_ALLOWLIST = {
    "--max-num-seqs",
    "--max-num-batched-tokens",
    "--block-size",
    "--swap-space",
    "--kv-cache-dtype",
    "--quantization",
    "--max-logprobs",
    "--disable-log-requests",
    "--disable-log-stats",
    "--enforce-eager",
    "--enable-prefix-caching",
    "--disable-custom-all-reduce",
    "--seed",
    "--revision",
    "--code-revision",
    "--tokenizer-revision",
    "--rope-scaling",
    "--rope-theta",
    "--tokenizer-mode",
    "--chat-template",
    "--response-role",
    "--guided-decoding-backend",
}

# ---------------- State ----------------

_instances: dict[str, VllmManager] = {}
_instance_counter = 0
_available_ports = list(range(VLLM_PORT_START, VLLM_PORT_END + 1))
_used_gpus: set[int] = set()

_download_state = {
    "status": "idle",  # idle, downloading, complete, error
    "repo_id": None,
    "error": None,
    "downloaded_bytes": 0,
    "total_bytes": 0,
}
_download_lock = asyncio.Lock()


def _persist() -> None:
    configs = {iid: inst.config for iid, inst in _instances.items() if inst.config}
    persistence.save_all(configs)


def _free_instance_resources(inst: VllmManager) -> None:
    """Release GPUs and the port used by this instance."""
    if not inst.config:
        return
    for gpu_id in inst.config.gpu_ids:
        _used_gpus.discard(gpu_id)
    port = inst.config.port
    if VLLM_PORT_START <= port <= VLLM_PORT_END and port not in _available_ports:
        _available_ports.append(port)
        _available_ports.sort()


def _on_instance_exit(instance_id: str, will_restart: bool) -> None:
    """Called by VllmManager when the vLLM process dies.

    If the manager is auto-restarting, we keep GPUs/port reserved. If it gave
    up, we drop the instance and free resources.
    """
    inst = _instances.get(instance_id)
    if not inst:
        return
    if will_restart:
        logger.info("[%s] process exited; manager is auto-restarting", instance_id)
        return
    _free_instance_resources(inst)
    _instances.pop(instance_id, None)
    _persist()
    logger.info("[%s] removed after terminal exit", instance_id)


# ---------------- Lifespan: reload persisted instances on startup ----------------


@asynccontextmanager
async def lifespan(_app: FastAPI):
    auth.log_startup_banner()
    await _restore_persisted()
    if cluster_hub is not None:
        cluster_hub.start()
    try:
        yield
    finally:
        if cluster_hub is not None:
            await cluster_hub.stop()
        # Best-effort graceful stop of everything on shutdown.
        await asyncio.gather(
            *(inst.stop() for inst in list(_instances.values())),
            return_exceptions=True,
        )


async def _restore_persisted() -> None:
    global _instance_counter
    saved = persistence.load_all()
    if not saved:
        return
    logger.info("Restoring %d persisted instance(s)", len(saved))
    for iid, cfg in saved.items():
        # Parse numeric suffix so auto-increment doesn't collide.
        suffix = iid.rsplit("-", 1)[-1]
        if suffix.isdigit():
            _instance_counter = max(_instance_counter, int(suffix))
        # Reserve resources optimistically; if start fails, roll back.
        if any(g in _used_gpus for g in cfg.gpu_ids):
            logger.warning("Skipping %s: GPUs already claimed by another persisted instance", iid)
            continue
        if cfg.port in _available_ports:
            _available_ports.remove(cfg.port)
        _used_gpus.update(cfg.gpu_ids)

        mgr = VllmManager(instance_id=iid, _on_exit=_on_instance_exit)
        _instances[iid] = mgr
        try:
            await mgr.start(cfg)
            logger.info("[%s] restored: %s on GPUs %s port %s", iid, cfg.model, cfg.gpu_ids, cfg.port)
        except Exception as e:
            logger.error("[%s] failed to restore: %s", iid, e)
            _free_instance_resources(mgr)
            _instances.pop(iid, None)


app = FastAPI(title="vLLM Admin", lifespan=lifespan)


@app.middleware("http")
async def _auth_mw(request: Request, call_next):
    return await auth.auth_middleware(request, call_next)


# ---------------- v0.4.0 control plane (opt-in via CONTROL_PLANE) ----------------

if CONTROL_PLANE:
    try:
        from admin.cluster.hub import ClusterHub, build_cluster_router

        cluster_hub = ClusterHub(
            config_path=str(MODELS_DIR / ".vllm-manager" / "cluster-config.json"),
            api_key=auth.ADMIN_API_KEY,
            auth_enabled=auth.AUTH_ENABLED,
            cookie_name=auth.COOKIE_NAME,   # so browser sessions authenticate the cluster API
        )
        app.include_router(build_cluster_router(cluster_hub))

        _STATIC = Path(__file__).parent / "static"

        @app.get("/cluster")
        def _cluster_page():
            return FileResponse(_STATIC / "cluster.html", headers={"Cache-Control": "no-cache"})

        @app.get("/cluster.js")
        def _cluster_js():
            return FileResponse(_STATIC / "cluster.js", media_type="application/javascript")

        @app.get("/cluster.css")
        def _cluster_css():
            return FileResponse(_STATIC / "cluster.css", media_type="text/css")

        @app.get("/serving.js")
        def _serving_js():
            return FileResponse(_STATIC / "serving.js", media_type="application/javascript")

        @app.get("/serving.css")
        def _serving_css():
            return FileResponse(_STATIC / "serving.css", media_type="text/css")

        @app.get("/join.sh")
        def _join_script():
            # Public on purpose: the script carries no secret — the operator supplies
            # the join token on the command line. See auth._PUBLIC_PATHS.
            return FileResponse(_STATIC / "join.sh", media_type="text/x-shellscript")

        logger.info("Control plane ENABLED: CCP WebSocket /api/ccp, UI /cluster")
    except Exception as _cp_err:  # noqa: BLE001
        logger.error("Control plane failed to initialise (%s); continuing without it", _cp_err)
        cluster_hub = None


# ---------------- Auth routes ----------------


class LoginRequest(BaseModel):
    api_key: str


@app.post("/api/auth/login")
def api_auth_login(req: LoginRequest):
    if not auth.AUTH_ENABLED:
        return {"status": "ok", "auth_enabled": False}
    if not auth.verify_key(req.api_key):
        return JSONResponse(status_code=401, content={"error": "Invalid API key"})
    resp = JSONResponse(content={"status": "ok"})
    resp.set_cookie(
        key=auth.COOKIE_NAME,
        value=req.api_key,
        max_age=auth.COOKIE_MAX_AGE,
        httponly=True,
        samesite="lax",
    )
    return resp


@app.post("/api/auth/logout")
def api_auth_logout():
    resp = JSONResponse(content={"status": "ok"})
    resp.delete_cookie(key=auth.COOKIE_NAME)
    return resp


@app.get("/api/auth/status")
def api_auth_status(request: Request):
    return {
        "auth_enabled": auth.AUTH_ENABLED,
        "authenticated": auth.is_authenticated(request),
    }


@app.get("/login")
def login_page():
    return FileResponse(
        Path(__file__).parent / "static" / "login.html",
        headers={"Cache-Control": "no-cache"},
    )


# ---------------- Helpers ----------------


def _safe_model_path(name: str) -> Optional[Path]:
    """Resolve a model name to a path under MODELS_DIR, or None if it escapes."""
    resolved = (MODELS_DIR / name).resolve()
    models_root = MODELS_DIR.resolve()
    if not resolved.is_relative_to(models_root):
        return None
    if resolved == models_root:
        return None
    return resolved


def _validate_extra_args(raw: str) -> tuple[Optional[list[str]], Optional[str]]:
    """Split raw extra args and reject flags not in the allowlist.

    Returns (parsed_args, error_msg). Exactly one is None.
    """
    if not raw.strip():
        return [], None
    try:
        tokens = shlex.split(raw)
    except ValueError as e:
        return None, f"Invalid extra args: {e}"
    for tok in tokens:
        if tok.startswith("--"):
            flag = tok.split("=", 1)[0]
            if flag not in _EXTRA_ARGS_ALLOWLIST:
                return None, f"Flag not allowed: {flag}"
    return tokens, None


def get_gpus() -> list[dict]:
    try:
        import pynvml
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        gpus = []
        for i in range(count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            name = pynvml.nvmlDeviceGetName(handle)
            if isinstance(name, bytes):
                name = name.decode()
            mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
            try:
                cc_major, cc_minor = pynvml.nvmlDeviceGetCudaComputeCapability(handle)
                compute_capability = f"{cc_major}.{cc_minor}"
            except Exception:
                compute_capability = "unknown"
            used_by = None
            for iid, inst in _instances.items():
                if inst.config and i in inst.config.gpu_ids and inst.state in (State.RUNNING, State.STARTING, State.RESTARTING):
                    used_by = iid
                    break
            gpus.append({
                "index": i,
                "name": name,
                "memory_total_mb": round(mem.total / 1024 / 1024),
                "memory_used_mb": round(mem.used / 1024 / 1024),
                "memory_free_mb": round(mem.free / 1024 / 1024),
                "compute_capability": compute_capability,
                "in_use": i in _used_gpus,
                "used_by": used_by,
            })
        pynvml.nvmlShutdown()
        return gpus
    except Exception as e:
        logger.error("GPU detection failed: %s", e)
        return [{"error": "GPU detection failed (see server logs)"}]


def list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    models = []
    for entry in sorted(MODELS_DIR.iterdir()):
        # Skip hidden dirs like .vllm-manager/
        if entry.name.startswith("."):
            continue
        if entry.is_dir():
            has_config = (entry / "config.json").exists()
            has_safetensors = any(entry.glob("*.safetensors"))
            has_bin = any(entry.glob("*.bin"))
            if has_config or has_safetensors or has_bin:
                models.append(entry.name)
    return models


@app.get("/api/gpus")
def api_gpus():
    return get_gpus()


@app.get("/api/models")
def api_models():
    return list_models()


def _get_num_hidden_layers(model_name: str) -> Optional[int]:
    model_path = _safe_model_path(model_name)
    if not model_path:
        return None
    config_file = model_path / "config.json"
    if not config_file.exists():
        return None
    try:
        with open(config_file) as f:
            config = json_mod.load(f)
        return config.get("num_hidden_layers")
    except Exception:
        return None


def _compute_pp_partition(num_layers: int, gpu_ids: list[int]) -> str:
    pp_size = len(gpu_ids)
    gpu_mem = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        for gid in gpu_ids:
            h = pynvml.nvmlDeviceGetHandleByIndex(gid)
            gpu_mem[gid] = pynvml.nvmlDeviceGetMemoryInfo(h).total
        pynvml.nvmlShutdown()
    except Exception:
        pass

    if not gpu_mem:
        base = num_layers // pp_size
        remainder = num_layers % pp_size
        parts = [base + (1 if i < remainder else 0) for i in range(pp_size)]
        return ",".join(str(p) for p in parts)

    total_mem = sum(gpu_mem.get(gid, 0) for gid in gpu_ids)
    if total_mem == 0:
        parts = [num_layers // pp_size] * pp_size
        parts[0] += num_layers - sum(parts)
        return ",".join(str(p) for p in parts)

    raw = [gpu_mem.get(gid, 0) / total_mem * num_layers for gid in gpu_ids]
    parts = [int(r) for r in raw]
    remainder = num_layers - sum(parts)
    fracs = [(raw[i] - parts[i], i) for i in range(pp_size)]
    fracs.sort(reverse=True)
    for j in range(remainder):
        parts[fracs[j][1]] += 1

    return ",".join(str(p) for p in parts)


@app.get("/api/model-info/{model_name}")
def api_model_info(model_name: str):
    model_path = _safe_model_path(model_name)
    if not model_path or not model_path.is_dir():
        return JSONResponse(status_code=404, content={"error": "Model not found"})
    num_layers = _get_num_hidden_layers(model_name)
    return {"model": model_name, "num_hidden_layers": num_layers}


async def _ray_cluster_info() -> dict:
    """Best-effort snapshot of the local Ray cluster via `ray list nodes`.

    Returns {nodes: [{ip, gpus, alive}], total_gpus, alive_nodes, error?}.
    Never raises — the cluster may be down or Ray not yet joined.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            "ray", "list", "nodes", "--format", "json", "--limit", "100",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=8)
        if proc.returncode != 0:
            return {"nodes": [], "total_gpus": 0, "alive_nodes": 0,
                    "error": (err.decode()[:300].strip() or "ray list nodes failed")}
        raw = json_mod.loads(out.decode() or "[]")
    except Exception as e:
        return {"nodes": [], "total_gpus": 0, "alive_nodes": 0, "error": str(e)}

    # Ray keeps records for dead nodes; a box that (re)joined shows up multiple
    # times. Collapse by IP so the UI shows one row per box (alive if any record
    # for that IP is alive), and count GPUs only for live boxes.
    by_ip: dict = {}
    for n in raw:
        alive = str(n.get("state", "")).upper() == "ALIVE"
        res = n.get("resources_total") or {}
        try:
            gpus = int(res.get("GPU", 0) or 0)
        except (TypeError, ValueError):
            gpus = 0
        ip = n.get("node_ip") or n.get("node_name") or "?"
        cur = by_ip.get(ip)
        if cur is None:
            by_ip[ip] = {"ip": ip, "gpus": gpus, "alive": alive}
        else:
            cur["alive"] = cur["alive"] or alive
            cur["gpus"] = max(cur["gpus"], gpus)

    nodes = list(by_ip.values())
    total = sum(n["gpus"] for n in nodes if n["alive"])
    alive_nodes = sum(1 for n in nodes if n["alive"])
    return {"nodes": nodes, "total_gpus": total, "alive_nodes": alive_nodes}


# Ray script (run as a subprocess) that probes each alive node for its total GPU
# memory. num_gpus=0 so it schedules even while an instance holds the GPUs; pynvml
# reads memory without reserving a device. Returns the driver IP + per-node totals.
_RAY_TOPO_SCRIPT = r"""
import json, ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
ray.init(address="auto", log_to_driver=False)
driver_ip = ray.util.get_node_ip_address()
alive = [n for n in ray.nodes() if n.get("Alive")]
@ray.remote(num_gpus=0)
def probe():
    import pynvml, ray as _r
    pynvml.nvmlInit()
    c = pynvml.nvmlDeviceGetCount()
    tot = sum(pynvml.nvmlDeviceGetMemoryInfo(pynvml.nvmlDeviceGetHandleByIndex(i)).total for i in range(c))
    pynvml.nvmlShutdown()
    return {"ip": _r.util.get_node_ip_address(), "gpu_count": c, "gpu_mem_total": int(tot)}
refs = [probe.options(scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=n["NodeID"], soft=False)).remote()
        for n in alive]
print(json.dumps({"driver_ip": driver_ip, "nodes": ray.get(refs)}))
"""


async def _cluster_gpu_topology() -> Optional[dict]:
    """Probe per-node total GPU memory across the Ray cluster (best effort)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "python3", "-c", _RAY_TOPO_SCRIPT,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        out, err = await asyncio.wait_for(proc.communicate(), timeout=30)
        if proc.returncode != 0:
            logger.warning("cluster GPU topology probe failed: %s", err.decode()[:300])
            return None
        # The script prints one JSON line; ignore any ray log noise on stdout.
        line = [l for l in out.decode().splitlines() if l.strip().startswith("{")]
        return json_mod.loads(line[-1]) if line else None
    except Exception as e:
        logger.warning("cluster GPU topology probe error: %s", e)
        return None


def _order_nodes_for_pp(topo: dict) -> list[dict]:
    """Order nodes the way vLLM assigns PP stages: driver node first, then others
    by (worker/GPU count, ip) — see ray_executor.sort_by_driver_then_worker_ip."""
    driver_ip = topo.get("driver_ip")
    nodes = [n for n in topo.get("nodes", []) if n.get("gpu_mem_total")]
    return sorted(nodes, key=lambda n: (0 if n["ip"] == driver_ip else 1,
                                        n.get("gpu_count", 0), n.get("ip", "")))


async def _compute_cluster_pp_partition(num_layers: int, pp_size: int) -> Optional[str]:
    """Memory-weighted PP layer split across nodes, in vLLM's PP-stage order.
    Bigger-memory nodes (e.g. the 32GB box) get proportionally more layers."""
    topo = await _cluster_gpu_topology()
    if not topo:
        return None
    ordered = _order_nodes_for_pp(topo)
    if len(ordered) != pp_size:
        logger.warning("PP auto-partition: probed %d nodes but pp_size=%d; using even split",
                       len(ordered), pp_size)
        return None
    mems = [n["gpu_mem_total"] for n in ordered]
    total_mem = sum(mems)
    if total_mem <= 0:
        return None
    raw = [m / total_mem * num_layers for m in mems]
    parts = [int(r) for r in raw]
    remainder = num_layers - sum(parts)
    for _, i in sorted(((raw[i] - parts[i], i) for i in range(len(raw))), reverse=True)[:remainder]:
        parts[i] += 1
    logger.info("PP auto-partition by GPU mem %s -> layers %s (nodes %s)",
                mems, parts, [n["ip"] for n in ordered])
    return ",".join(str(p) for p in parts)


@app.get("/api/cluster")
async def api_cluster():
    if not CLUSTER_MODE:
        return {"enabled": False}
    info = await _ray_cluster_info()
    return {"enabled": True, **info}


@app.get("/api/status")
def api_status():
    return {"instances": [inst.get_status() for inst in _instances.values()]}


class StartRequest(BaseModel):
    model: str
    gpu_ids: list[int]
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    model_impl: str = "auto"
    served_model_name: Optional[str] = None
    language_model_only: bool = False
    pipeline_parallel_size: int = 1
    pp_layer_partition: Optional[str] = None
    enable_tool_use: bool = False
    tool_call_parser: Optional[str] = None
    extra_args: str = ""
    auto_restart: bool = True
    # When True, span the Ray cluster (remote GPUs on other boxes). gpu_ids /
    # pipeline_parallel_size from the form are ignored; the layout is derived
    # from the cluster shape (TP within a node, PP across nodes).
    use_cluster: bool = False


async def _start_cluster_instance(req: StartRequest):
    """Launch a vLLM instance that spans the whole Ray cluster (remote GPUs).

    Layout is homogeneous: tensor-parallel within each node, pipeline-parallel
    across nodes. Ray schedules the workers, so we don't pin GPU indices; we do
    reserve all local GPUs so no single-node instance collides with placement.
    """
    global _instance_counter

    if not CLUSTER_MODE:
        return JSONResponse(status_code=400, content={
            "error": "Cluster mode is not enabled on this container. Set CLUSTER_MODE=true and restart."
        })
    if req.dtype not in ALLOWED_DTYPES:
        return JSONResponse(status_code=400, content={"error": f"Invalid dtype: {req.dtype}"})
    if req.model_impl not in ALLOWED_MODEL_IMPLS:
        return JSONResponse(status_code=400, content={"error": f"Invalid model_impl: {req.model_impl}"})
    if req.enable_tool_use and not req.tool_call_parser:
        return JSONResponse(status_code=400, content={"error": "tool_call_parser is required when enable_tool_use is true"})
    if req.tool_call_parser and req.tool_call_parser not in ALLOWED_TOOL_PARSERS:
        return JSONResponse(status_code=400, content={"error": f"Invalid tool_call_parser: {req.tool_call_parser}"})

    model_path = _safe_model_path(req.model)
    if not model_path or not model_path.is_dir():
        return JSONResponse(status_code=400, content={"error": "Invalid model name"})

    info = await _ray_cluster_info()
    if info.get("error"):
        return JSONResponse(status_code=502, content={"error": f"Ray cluster unavailable: {info['error']}"})
    num_nodes = info.get("alive_nodes", 0)
    total_gpus = info.get("total_gpus", 0)
    if num_nodes < 2:
        return JSONResponse(status_code=409, content={
            "error": f"Need at least 2 Ray nodes for remote GPUs; {num_nodes} joined. Start a worker on the other box."
        })
    if total_gpus < 2 or total_gpus % num_nodes != 0:
        return JSONResponse(status_code=400, content={
            "error": f"Cluster has {total_gpus} GPU(s) across {num_nodes} node(s); need an even split. "
                     "Balance the boxes or launch per-node instances instead."
        })
    pp_size = num_nodes
    tp_size = total_gpus // num_nodes

    # A cluster instance uses the whole local box; refuse if local GPUs are busy.
    if _used_gpus:
        return JSONResponse(status_code=409, content={
            "error": f"Local GPUs {sorted(_used_gpus)} are in use. Stop local instances before launching a cluster instance."
        })
    if not _available_ports:
        return JSONResponse(status_code=409, content={"error": f"No available ports (max {MAX_INSTANCES} instances)"})

    extra_args, err = _validate_extra_args(req.extra_args)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    # PP layer partition across nodes. A manual value wins; otherwise auto-compute
    # a memory-weighted split so bigger-memory nodes (e.g. the 32GB box) take more
    # layers than the 16GB box — an even split would OOM the smaller GPUs.
    pp_layer_partition = req.pp_layer_partition or None
    if pp_layer_partition:
        try:
            parts = [int(x) for x in pp_layer_partition.split(",")]
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "PP layer partition must be comma-separated integers"})
        if len(parts) != pp_size:
            return JSONResponse(status_code=400, content={
                "error": f"PP layer partition needs {pp_size} values (one per node), got {len(parts)}"
            })
        num_layers = _get_num_hidden_layers(req.model)
        if num_layers and sum(parts) != num_layers:
            return JSONResponse(status_code=400, content={
                "error": f"PP layer partition sums to {sum(parts)} but model has {num_layers} layers"
            })
    else:
        num_layers = _get_num_hidden_layers(req.model)
        if num_layers:
            pp_layer_partition = await _compute_cluster_pp_partition(num_layers, pp_size)
            if pp_layer_partition:
                logger.info("Auto cross-node PP partition: %s (model has %d layers)",
                            pp_layer_partition, num_layers)

    port = _available_ports.pop(0)
    _instance_counter += 1
    instance_id = f"instance-{_instance_counter}"

    local_gpu_ids = [g["index"] for g in get_gpus() if "index" in g]

    config = VllmConfig(
        model=str(model_path),
        gpu_ids=local_gpu_ids,
        tensor_parallel_size=tp_size,
        port=port,
        gpu_memory_utilization=req.gpu_memory_utilization,
        max_model_len=req.max_model_len,
        dtype=req.dtype,
        model_impl=req.model_impl,
        served_model_name=req.served_model_name or None,
        language_model_only=req.language_model_only,
        pipeline_parallel_size=pp_size,
        pp_layer_partition=pp_layer_partition,
        enable_tool_use=req.enable_tool_use,
        tool_call_parser=req.tool_call_parser,
        extra_args=extra_args or [],
        auto_restart=req.auto_restart,
        use_cluster=True,
    )

    mgr = VllmManager(instance_id=instance_id, _on_exit=_on_instance_exit)
    _instances[instance_id] = mgr
    _used_gpus.update(local_gpu_ids)

    try:
        await mgr.start(config)
        _persist()
        return {"status": "starting", "id": instance_id, "port": port, "pid": mgr.pid,
                "cluster": {"nodes": num_nodes, "tensor_parallel_size": tp_size, "pipeline_parallel_size": pp_size}}
    except RuntimeError as e:
        _free_instance_resources(mgr)
        _instances.pop(instance_id, None)
        return JSONResponse(status_code=409, content={"error": str(e)})


@app.post("/api/start")
async def api_start(req: StartRequest):
    global _instance_counter

    if req.use_cluster:
        return await _start_cluster_instance(req)

    if req.dtype not in ALLOWED_DTYPES:
        return JSONResponse(status_code=400, content={"error": f"Invalid dtype: {req.dtype}"})
    if req.model_impl not in ALLOWED_MODEL_IMPLS:
        return JSONResponse(status_code=400, content={"error": f"Invalid model_impl: {req.model_impl}"})
    if not req.gpu_ids:
        return JSONResponse(status_code=400, content={"error": "At least one GPU must be selected"})
    if req.pipeline_parallel_size > len(req.gpu_ids):
        return JSONResponse(status_code=400, content={"error": "Pipeline parallel size cannot exceed number of GPUs"})
    if len(req.gpu_ids) % req.pipeline_parallel_size != 0:
        return JSONResponse(status_code=400, content={
            "error": f"Number of GPUs ({len(req.gpu_ids)}) must be divisible by pipeline parallel size ({req.pipeline_parallel_size})"
        })
    if req.pp_layer_partition:
        try:
            parts = [int(x) for x in req.pp_layer_partition.split(",")]
            if len(parts) != req.pipeline_parallel_size:
                return JSONResponse(status_code=400, content={
                    "error": f"PP layer partition has {len(parts)} values but pipeline_parallel_size is {req.pipeline_parallel_size}"
                })
            num_layers = _get_num_hidden_layers(req.model)
            if num_layers and sum(parts) != num_layers:
                return JSONResponse(status_code=400, content={
                    "error": f"PP layer partition sums to {sum(parts)} but model has {num_layers} layers"
                })
        except ValueError:
            return JSONResponse(status_code=400, content={"error": "PP layer partition must be comma-separated integers (e.g. 14,26)"})
    if req.enable_tool_use and not req.tool_call_parser:
        return JSONResponse(status_code=400, content={"error": "tool_call_parser is required when enable_tool_use is true"})
    if req.tool_call_parser and req.tool_call_parser not in ALLOWED_TOOL_PARSERS:
        return JSONResponse(status_code=400, content={"error": f"Invalid tool_call_parser: {req.tool_call_parser}"})
    model_path = _safe_model_path(req.model)
    if not model_path or not model_path.is_dir():
        return JSONResponse(status_code=400, content={"error": "Invalid model name"})

    overlap = set(req.gpu_ids) & _used_gpus
    if overlap:
        return JSONResponse(status_code=409, content={"error": f"GPUs already in use: {sorted(overlap)}"})

    if not _available_ports:
        return JSONResponse(status_code=409, content={"error": f"No available ports (max {MAX_INSTANCES} instances)"})

    extra_args, err = _validate_extra_args(req.extra_args)
    if err:
        return JSONResponse(status_code=400, content={"error": err})

    port = _available_ports.pop(0)

    _instance_counter += 1
    instance_id = f"instance-{_instance_counter}"

    # Sort GPUs by total memory descending (largest first) for PP layer placement
    gpu_mem = {}
    try:
        import pynvml
        pynvml.nvmlInit()
        for gid in req.gpu_ids:
            h = pynvml.nvmlDeviceGetHandleByIndex(gid)
            gpu_mem[gid] = pynvml.nvmlDeviceGetMemoryInfo(h).total
        pynvml.nvmlShutdown()
    except Exception:
        pass
    sorted_gpu_ids = sorted(req.gpu_ids, key=lambda g: gpu_mem.get(g, 0), reverse=True)

    pp_layer_partition = req.pp_layer_partition or None
    if req.pipeline_parallel_size > 1 and not pp_layer_partition:
        num_layers = _get_num_hidden_layers(req.model)
        if num_layers:
            pp_layer_partition = _compute_pp_partition(num_layers, sorted_gpu_ids)
            logger.info("Auto-computed PP layer partition: %s (model has %d layers)", pp_layer_partition, num_layers)

    config = VllmConfig(
        model=str(model_path),
        gpu_ids=sorted_gpu_ids,
        tensor_parallel_size=len(req.gpu_ids) // req.pipeline_parallel_size,
        port=port,
        gpu_memory_utilization=req.gpu_memory_utilization,
        max_model_len=req.max_model_len,
        dtype=req.dtype,
        model_impl=req.model_impl,
        served_model_name=req.served_model_name or None,
        language_model_only=req.language_model_only,
        pipeline_parallel_size=req.pipeline_parallel_size,
        pp_layer_partition=pp_layer_partition,
        enable_tool_use=req.enable_tool_use,
        tool_call_parser=req.tool_call_parser,
        extra_args=extra_args or [],
        auto_restart=req.auto_restart,
    )

    mgr = VllmManager(instance_id=instance_id, _on_exit=_on_instance_exit)
    _instances[instance_id] = mgr
    _used_gpus.update(req.gpu_ids)

    try:
        await mgr.start(config)
        _persist()
        return {"status": "starting", "id": instance_id, "port": port, "pid": mgr.pid}
    except RuntimeError as e:
        _free_instance_resources(mgr)
        _instances.pop(instance_id, None)
        return JSONResponse(status_code=409, content={"error": str(e)})


class StopRequest(BaseModel):
    instance_id: str


@app.post("/api/stop")
async def api_stop(req: StopRequest):
    mgr = _instances.get(req.instance_id)
    if not mgr:
        return JSONResponse(status_code=404, content={"error": f"Instance {req.instance_id} not found"})

    await mgr.stop()
    _free_instance_resources(mgr)
    _instances.pop(req.instance_id, None)
    _persist()
    return {"status": "stopped", "instance_id": req.instance_id}


class DeleteModelRequest(BaseModel):
    model: str


@app.post("/api/models/delete")
def api_delete_model(req: DeleteModelRequest):
    model_path = _safe_model_path(req.model)
    if not model_path or not model_path.exists() or not model_path.is_dir():
        return JSONResponse(status_code=400, content={"error": "Invalid model name"})

    full_path = str(model_path)
    for iid, inst in _instances.items():
        if inst.config and inst.config.model == full_path and inst.state in (State.RUNNING, State.STARTING, State.RESTARTING):
            return JSONResponse(
                status_code=409,
                content={"error": f"Model is in use by {iid}. Stop the instance first."},
            )

    shutil.rmtree(model_path)
    logger.info("Deleted model: %s", req.model)
    return {"status": "deleted", "model": req.model}


class DownloadRequest(BaseModel):
    repo_id: str


def _do_download(repo_id: str) -> None:
    from huggingface_hub import snapshot_download
    from tqdm.auto import tqdm as tqdm_auto

    class DownloadProgress(tqdm_auto):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if self.total:
                _download_state["total_bytes"] = self.total

        def update(self, n=1):
            super().update(n)
            _download_state["downloaded_bytes"] = self.n
            if self.total:
                _download_state["total_bytes"] = self.total

    local_name = repo_id.split("/")[-1]
    local_dir = _safe_model_path(local_name)
    if not local_dir:
        raise ValueError(f"Invalid repo ID: {repo_id}")

    hf_token = os.getenv("HF_TOKEN") or None
    snapshot_download(
        repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        tqdm_class=DownloadProgress,
        token=hf_token,
    )


async def _run_download(repo_id: str) -> None:
    _download_state["status"] = "downloading"
    _download_state["repo_id"] = repo_id
    _download_state["error"] = None
    _download_state["downloaded_bytes"] = 0
    _download_state["total_bytes"] = 0
    try:
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_download, repo_id)
        _download_state["status"] = "complete"
        logger.info("Download complete: %s", repo_id)
    except Exception as e:
        _download_state["status"] = "error"
        _download_state["error"] = str(e)
        logger.error("Download failed: %s", e)


@app.post("/api/download")
async def api_download(req: DownloadRequest):
    async with _download_lock:
        if _download_state["status"] == "downloading":
            return JSONResponse(
                status_code=409,
                content={"error": f"Already downloading {_download_state['repo_id']}"},
            )
        # Flip state inside the lock so a concurrent call sees it.
        _download_state["status"] = "downloading"
        _download_state["repo_id"] = req.repo_id
        _download_state["error"] = None
        _download_state["downloaded_bytes"] = 0
        _download_state["total_bytes"] = 0
    asyncio.create_task(_run_download(req.repo_id))
    return {"status": "downloading", "repo_id": req.repo_id}


@app.get("/api/download/status")
def api_download_status():
    return _download_state


class ChatRequest(BaseModel):
    instance_id: str
    messages: list[dict]
    tools: Optional[list[dict]] = None
    temperature: float = 0.7
    max_tokens: int = 1024


@app.post("/api/chat")
async def api_chat(req: ChatRequest):
    mgr = _instances.get(req.instance_id)
    if not mgr:
        return JSONResponse(status_code=404, content={"error": "Instance not found"})
    if mgr.state != State.RUNNING:
        return JSONResponse(status_code=409, content={"error": f"Instance is {mgr.state.value}, not running"})

    port = mgr.config.port
    model = mgr.config.served_model_name or mgr.config.model

    payload = {
        "model": model,
        "messages": req.messages,
        "temperature": req.temperature,
        "max_tokens": req.max_tokens,
    }
    if req.tools:
        payload["tools"] = req.tools

    import httpx
    async with httpx.AsyncClient(timeout=120) as client:
        try:
            resp = await client.post(
                f"http://localhost:{port}/v1/chat/completions",
                json=payload,
            )
            return JSONResponse(content=resp.json(), status_code=resp.status_code)
        except httpx.RequestError as e:
            return JSONResponse(status_code=502, content={"error": f"Failed to reach vLLM: {e}"})


@app.get("/healthz")
def healthz():
    return Response(content="ok", media_type="text/plain")


@app.get("/")
def root():
    """The landing page.

    With the control plane on, "/" IS the tabbed cluster UI (Summary | Serving |
    Workers | Storage | Configuration) — the old single-node page is redundant
    because Serving absorbed its model library, downloads and launch, and does
    so cluster-wide. With CONTROL_PLANE off, "/" is the unchanged v0.3.0 page.
    The old page stays reachable at /legacy either way (it still has the chat
    tester, which the tabs do not cover yet).
    """
    page = "cluster.html" if (CONTROL_PLANE and cluster_hub is not None) else "index.html"
    return FileResponse(
        Path(__file__).parent / "static" / page,
        headers={"Cache-Control": "no-cache"},
    )


@app.get("/legacy")
def legacy_ui():
    """The v0.3.0 single-node UI, always available (retains the chat tester)."""
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-cache"},
    )
