import asyncio
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
    disable_image_processor: bool = False
    extra_args: str = ""


@app.post("/api/start")
async def api_start(req: StartRequest):
    global _instance_counter

    # Validate inputs
    if req.dtype not in ALLOWED_DTYPES:
        return JSONResponse(status_code=400, content={"error": f"Invalid dtype: {req.dtype}"})
    if req.model_impl not in ALLOWED_MODEL_IMPLS:
        return JSONResponse(status_code=400, content={"error": f"Invalid model_impl: {req.model_impl}"})
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

    config = VllmConfig(
        model=str(model_path),
        gpu_ids=req.gpu_ids,
        tensor_parallel_size=len(req.gpu_ids),
        port=port,
        gpu_memory_utilization=req.gpu_memory_utilization,
        max_model_len=req.max_model_len,
        dtype=req.dtype,
        model_impl=req.model_impl,
        disable_image_processor=req.disable_image_processor,
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


@app.get("/")
def root():
    return FileResponse(
        Path(__file__).parent / "static" / "index.html",
        headers={"Cache-Control": "no-cache"},
    )
