import asyncio
import logging
import os
import signal
import subprocess
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, Optional

import httpx

logger = logging.getLogger(__name__)


class State(str, Enum):
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    ERROR = "error"


@dataclass
class VllmConfig:
    model: str
    gpu_ids: list[int]
    tensor_parallel_size: int
    port: int = 8001
    gpu_memory_utilization: float = 0.90
    max_model_len: Optional[int] = None
    dtype: str = "auto"
    model_impl: str = "auto"
    language_model_only: bool = False
    served_model_name: Optional[str] = None
    pipeline_parallel_size: int = 1
    pp_layer_partition: Optional[str] = None
    enable_tool_use: bool = False
    tool_call_parser: Optional[str] = None
    extra_args: list[str] = field(default_factory=list)


@dataclass
class VllmManager:
    instance_id: str = ""
    state: State = State.STOPPED
    config: Optional[VllmConfig] = None
    process: Optional[subprocess.Popen] = None
    pid: Optional[int] = None
    logs: list[str] = field(default_factory=list)
    _cmd: list[str] = field(default_factory=list)
    _log_task: Optional[asyncio.Task] = None
    _health_task: Optional[asyncio.Task] = None
    _on_exit: Optional[Callable] = None

    def _build_cmd(self, config: VllmConfig) -> list[str]:
        cmd = [
            "python3", "-m", "vllm.entrypoints.openai.api_server",
            "--model", config.model,
        ]
        if config.language_model_only:
            cmd.append("--language-model-only")
        if config.enable_tool_use:
            cmd.append("--enable-auto-tool-choice")
            if config.tool_call_parser:
                cmd.extend(["--tool-call-parser", config.tool_call_parser])
        if config.pipeline_parallel_size > 1:
            cmd.extend(["--pipeline-parallel-size", str(config.pipeline_parallel_size)])
        cmd.extend([
            "--tensor-parallel-size", str(config.tensor_parallel_size),
            "--gpu-memory-utilization", str(config.gpu_memory_utilization),
            "--dtype", config.dtype,
            "--host", "0.0.0.0",
            "--port", str(config.port),
            "--model-impl", config.model_impl,
        ])
        if config.served_model_name:
            cmd.extend(["--served-model-name", config.served_model_name])
        if config.max_model_len is not None:
            cmd.extend(["--max-model-len", str(config.max_model_len)])
        if config.extra_args:
            cmd.extend(config.extra_args)
        return cmd

    def _build_env(self, config: VllmConfig) -> dict[str, str]:
        env = os.environ.copy()
        env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        env["CUDA_VISIBLE_DEVICES"] = ",".join(str(g) for g in config.gpu_ids)
        if config.pp_layer_partition:
            env["VLLM_PP_LAYER_PARTITION"] = config.pp_layer_partition
        return env

    async def start(self, config: VllmConfig) -> None:
        if self.state in (State.RUNNING, State.STARTING):
            raise RuntimeError(f"vLLM is already {self.state.value}")

        self.config = config
        self.state = State.STARTING
        self.logs = []

        cmd = self._build_cmd(config)
        self._cmd = cmd
        env = self._build_env(config)

        logger.info("[%s] Starting vLLM on port %s: %s", self.instance_id, config.port, " ".join(cmd))
        logger.info("[%s] CUDA_VISIBLE_DEVICES=%s", self.instance_id, env["CUDA_VISIBLE_DEVICES"])

        self.process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        self.pid = self.process.pid
        self._log_task = asyncio.create_task(self._read_logs())
        self._health_task = asyncio.create_task(self._poll_health())

    async def _read_logs(self) -> None:
        loop = asyncio.get_event_loop()
        try:
            while self.process and self.process.stdout:
                line = await loop.run_in_executor(
                    None, self.process.stdout.readline
                )
                if not line:
                    break
                line = line.rstrip("\n")
                self.logs.append(line)
                # Keep last 500 lines
                if len(self.logs) > 500:
                    self.logs = self.logs[-500:]
        except Exception:
            pass

    async def _poll_health(self) -> None:
        port = self.config.port if self.config else 8001
        async with httpx.AsyncClient() as client:
            while True:
                await asyncio.sleep(3)
                if self.process is None:
                    return
                # Check if process exited
                ret = self.process.poll()
                if ret is not None:
                    self.state = State.ERROR if ret != 0 else State.STOPPED
                    self.process = None
                    self.pid = None
                    if self._on_exit:
                        self._on_exit(self.instance_id)
                    return
                # Check health endpoint
                if self.state == State.STARTING:
                    try:
                        resp = await client.get(
                            f"http://localhost:{port}/health", timeout=2
                        )
                        if resp.status_code == 200:
                            self.state = State.RUNNING
                            logger.info("[%s] vLLM is healthy and running on port %s", self.instance_id, port)
                    except httpx.RequestError:
                        pass

    async def stop(self) -> None:
        if self.process is None:
            self.state = State.STOPPED
            return

        logger.info("[%s] Stopping vLLM (PID %s)", self.instance_id, self.pid)
        self.process.send_signal(signal.SIGTERM)

        # Wait up to 15 seconds for graceful shutdown
        for _ in range(30):
            if self.process.poll() is not None:
                break
            await asyncio.sleep(0.5)
        else:
            logger.warning("[%s] vLLM did not exit gracefully, sending SIGKILL", self.instance_id)
            self.process.kill()
            self.process.wait()

        self.process = None
        self.pid = None
        self.state = State.STOPPED

        if self._log_task:
            self._log_task.cancel()
        if self._health_task:
            self._health_task.cancel()

    def get_status(self) -> dict:
        return {
            "id": self.instance_id,
            "state": self.state.value,
            "model": self.config.model if self.config else None,
            "gpu_ids": self.config.gpu_ids if self.config else [],
            "pid": self.pid,
            "port": self.config.port if self.config else None,
            "dtype": self.config.dtype if self.config else None,
            "gpu_memory_utilization": self.config.gpu_memory_utilization if self.config else None,
            "max_model_len": self.config.max_model_len if self.config else None,
            "tensor_parallel_size": self.config.tensor_parallel_size if self.config else None,
            "model_impl": self.config.model_impl if self.config else None,
            "pipeline_parallel_size": self.config.pipeline_parallel_size if self.config else 1,
            "pp_layer_partition": self.config.pp_layer_partition if self.config else None,
            "enable_tool_use": self.config.enable_tool_use if self.config else False,
            "tool_call_parser": self.config.tool_call_parser if self.config else None,
            "cmd": " ".join(self._cmd) if self._cmd else None,
            "cuda_visible_devices": ",".join(str(g) for g in self.config.gpu_ids) if self.config else None,
            "logs": self.logs[-100:],
        }
