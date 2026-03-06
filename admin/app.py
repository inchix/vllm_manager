import asyncio
import json as json_mod
import logging
import os
import shlex
import shutil
from pathlib import Path
from typing import Optional

from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

from admin.vllm_manager import State, VllmConfig, VllmManager

logger = logging.getLogger(__name__)

app = FastAPI(title="vLLM Admin")

# Multi-instance state
_instances: dict[str, VllmManager] = {}
_instance_counter = 0
_available_ports = list(range(8001, 8011))
_used_gpus: set[int] = set()

# Download state
_download_state = {
    "status": "idle",  # idle, downloading, complete, error
    "repo_id": None,
    "error": None,
    "downloaded_bytes": 0,
    "total_bytes": 0,
}

MODELS_DIR = Path(os.getenv("MODELS_DIR", "/models"))

ALLOWED_DTYPES = {"auto", "float16", "bfloat16", "float32"}
ALLOWED_MODEL_IMPLS = {"auto", "transformers", "vllm"}
ALLOWED_TOOL_PARSERS = {
    "hermes", "llama3_json", "llama4_json", "mistral",
    "qwen3_xml", "qwen3_coder", "deepseek_v3", "pythonic", "openai",
}


def _safe_model_path(name: str) -> Optional[Path]:
    """Resolve model name to a path within MODELS_DIR, or None if invalid."""
    resolved = (MODELS_DIR / name).resolve()
    if not resolved.is_relative_to(MODELS_DIR.resolve()):
        return None
    return resolved


def _on_instance_exit(instance_id: str) -> None:
    """Called when a vLLM process exits unexpectedly."""
    inst = _instances.get(instance_id)
    if inst and inst.config:
        for gpu_id in inst.config.gpu_ids:
            _used_gpus.discard(gpu_id)
        if inst.config.port in range(8001, 8011):
            _available_ports.append(inst.config.port)
            _available_ports.sort()
        logger.info("[%s] Process exited, freed GPUs %s and port %s",
                    instance_id, inst.config.gpu_ids, inst.config.port)


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
            # Find which instance is using this GPU
            used_by = None
            for iid, inst in _instances.items():
                if inst.config and i in inst.config.gpu_ids and inst.state in (State.RUNNING, State.STARTING):
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
        return [{"error": str(e)}]


def list_models() -> list[str]:
    if not MODELS_DIR.exists():
        return []
    models = []
    for entry in sorted(MODELS_DIR.iterdir()):
        if entry.is_dir():
            # Check for model files (config.json or tokenizer files)
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
    """Read num_hidden_layers from a model's config.json."""
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
    """Compute PP layer partition proportional to GPU memory."""
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
        # Equal split if we can't read GPU memory
        base = num_layers // pp_size
        remainder = num_layers % pp_size
        parts = [base + (1 if i < remainder else 0) for i in range(pp_size)]
        return ",".join(str(p) for p in parts)

    total_mem = sum(gpu_mem.get(gid, 0) for gid in gpu_ids)
    if total_mem == 0:
        parts = [num_layers // pp_size] * pp_size
        parts[0] += num_layers - sum(parts)
        return ",".join(str(p) for p in parts)

    # Distribute layers proportionally to memory
    raw = [gpu_mem.get(gid, 0) / total_mem * num_layers for gid in gpu_ids]
    parts = [int(r) for r in raw]
    # Distribute remaining layers to GPUs with largest fractional parts
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


@app.get("/api/status")
def api_status():
    instances = []
    for iid, inst in _instances.items():
        instances.append(inst.get_status())
    return {"instances": instances}


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


@app.post("/api/start")
async def api_start(req: StartRequest):
    global _instance_counter

    # Validate inputs
    if req.dtype not in ALLOWED_DTYPES:
        return JSONResponse(status_code=400, content={"error": f"Invalid dtype: {req.dtype}"})
    if req.model_impl not in ALLOWED_MODEL_IMPLS:
        return JSONResponse(status_code=400, content={"error": f"Invalid model_impl: {req.model_impl}"})
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
    if req.tool_call_parser and req.tool_call_parser not in ALLOWED_TOOL_PARSERS:
        return JSONResponse(status_code=400, content={"error": f"Invalid tool_call_parser: {req.tool_call_parser}"})
    model_path = _safe_model_path(req.model)
    if not model_path or not model_path.is_dir():
        return JSONResponse(status_code=400, content={"error": "Invalid model name"})

    # Check GPU conflicts
    overlap = set(req.gpu_ids) & _used_gpus
    if overlap:
        return JSONResponse(
            status_code=409,
            content={"error": f"GPUs already in use: {sorted(overlap)}"},
        )

    # Allocate port
    if not _available_ports:
        return JSONResponse(
            status_code=409,
            content={"error": "No available ports (max 10 instances)"},
        )
    port = _available_ports.pop(0)

    # Create instance
    _instance_counter += 1
    instance_id = f"instance-{_instance_counter}"

    # Parse and validate extra args
    extra_args = []
    if req.extra_args.strip():
        try:
            extra_args = shlex.split(req.extra_args)
        except ValueError as e:
            _available_ports.append(port)
            _available_ports.sort()
            return JSONResponse(status_code=400, content={"error": f"Invalid extra args: {e}"})

    # Sort GPUs by total memory descending (largest first)
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

    # Auto-compute PP layer partition if PP > 1 and not specified
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
        extra_args=extra_args,
    )

    mgr = VllmManager(instance_id=instance_id, _on_exit=_on_instance_exit)
    _instances[instance_id] = mgr
    _used_gpus.update(req.gpu_ids)

    try:
        await mgr.start(config)
        return {"status": "starting", "id": instance_id, "port": port, "pid": mgr.pid}
    except RuntimeError as e:
        # Rollback
        _used_gpus.difference_update(req.gpu_ids)
        _available_ports.append(port)
        _available_ports.sort()
        del _instances[instance_id]
        return JSONResponse(status_code=409, content={"error": str(e)})


class StopRequest(BaseModel):
    instance_id: str


@app.post("/api/stop")
async def api_stop(req: StopRequest):
    mgr = _instances.get(req.instance_id)
    if not mgr:
        return JSONResponse(status_code=404, content={"error": f"Instance {req.instance_id} not found"})

    config = mgr.config
    await mgr.stop()

    # Free resources
    if config:
        for gpu_id in config.gpu_ids:
            _used_gpus.discard(gpu_id)
        if config.port in range(8001, 8011):
            _available_ports.append(config.port)
            _available_ports.sort()

    del _instances[req.instance_id]
    return {"status": "stopped", "instance_id": req.instance_id}


class DeleteModelRequest(BaseModel):
    model: str


@app.post("/api/models/delete")
def api_delete_model(req: DeleteModelRequest):
    model_path = _safe_model_path(req.model)
    if not model_path or not model_path.exists() or not model_path.is_dir():
        return JSONResponse(status_code=400, content={"error": "Invalid model name"})

    # Check if model is in use by a running instance
    full_path = str(model_path)
    for iid, inst in _instances.items():
        if inst.config and inst.config.model == full_path and inst.state in (State.RUNNING, State.STARTING):
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
    """Blocking download — runs in a thread via asyncio."""
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

    snapshot_download(
        repo_id,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        tqdm_class=DownloadProgress,
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
    if _download_state["status"] == "downloading":
        return JSONResponse(
            status_code=409,
            content={"error": f"Already downloading {_download_state['repo_id']}"},
        )
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


@app.get("/")
def root():
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-cache"},
    )
