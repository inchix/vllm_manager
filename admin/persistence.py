"""Persist instance configs to disk so instances survive container restarts."""
import json
import logging
import os
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Optional

from admin.vllm_manager import VllmConfig

logger = logging.getLogger(__name__)


def _state_dir() -> Path:
    base = Path(os.getenv("MODELS_DIR", "/models"))
    return base / ".vllm-manager"


def _state_file() -> Path:
    return _state_dir() / "instances.json"


def _atomic_write(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".instances-", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w") as f:
            f.write(data)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def save_all(configs: dict[str, VllmConfig]) -> None:
    """Write every live instance config to disk.

    Key is instance_id. Called on start/stop; crashes that auto-restart don't
    rewrite because the config hasn't changed.
    """
    try:
        payload = {
            "version": 1,
            "instances": {iid: asdict(cfg) for iid, cfg in configs.items()},
        }
        _atomic_write(_state_file(), json.dumps(payload, indent=2))
    except Exception as e:
        logger.error("Failed to persist instance state: %s", e)


def load_all() -> dict[str, VllmConfig]:
    path = _state_file()
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text())
    except Exception as e:
        logger.error("Failed to read persisted state (%s): %s", path, e)
        return {}

    if not isinstance(payload, dict) or payload.get("version") != 1:
        logger.warning("Persisted state has unknown format; ignoring")
        return {}

    out: dict[str, VllmConfig] = {}
    for iid, raw in (payload.get("instances") or {}).items():
        try:
            out[iid] = VllmConfig(**raw)
        except TypeError as e:
            logger.warning("Skipping malformed persisted instance %s: %s", iid, e)
    return out


def clear() -> None:
    try:
        _state_file().unlink(missing_ok=True)
    except Exception as e:
        logger.error("Failed to clear persisted state: %s", e)
