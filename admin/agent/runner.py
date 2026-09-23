"""Runner — executes CCP commands locally on a node.

Real implementations for the parts we can validate now (config/power-cap, storage mount,
modelfsd serve, model sync, GPU telemetry) and a coherent Ray+vLLM launch path for
``ensure_replica`` that mirrors the v0.3.0 command line (docs/04). The multi-node launch path is
implemented but has not yet been hardware-validated through the agent (that is Phase 4 in
docs/06); every command reports a structured ok/error result so the admin sees the truth.

stdlib + subprocess only; Python 3.9+.
"""
from __future__ import annotations

import os
import shlex
import signal
import subprocess
import time
from typing import Optional


def _run(cmd: list, timeout: int = 60) -> tuple:
    """Run a command, return (rc, stdout, stderr). Never raises on non-zero."""
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except FileNotFoundError as exc:
        return 127, "", str(exc)
    except subprocess.TimeoutExpired:
        return 124, "", "timeout"


class Runner:
    def __init__(self):
        self._replicas: dict = {}     # replica_id -> {procs: [Popen], role, port, state}
        self._shares: dict = {}       # export path -> {proc, endpoint, port}
        self._mounts: dict = {}       # path -> {source, ok}

    # -- GPU telemetry ----------------------------------------------------
    def _telemetry_via_pynvml(self):
        """Live per-GPU stats via NVML (works in the hardened container). None if
        NVML is unavailable, so the caller falls back to nvidia-smi."""
        try:
            import pynvml
            pynvml.nvmlInit()
        except Exception:
            return None
        gpus = []
        try:
            for i in range(pynvml.nvmlDeviceGetCount()):
                h = pynvml.nvmlDeviceGetHandleByIndex(i)
                g = {"index": i}
                try:
                    g["util"] = float(pynvml.nvmlDeviceGetUtilizationRates(h).gpu)
                except Exception:
                    g["util"] = None
                try:
                    m = pynvml.nvmlDeviceGetMemoryInfo(h)
                    g["mem_used"] = round(m.used / (1024 * 1024))
                    g["mem_total"] = round(m.total / (1024 * 1024))
                except Exception:
                    pass
                try:
                    g["temp"] = float(pynvml.nvmlDeviceGetTemperature(
                        h, pynvml.NVML_TEMPERATURE_GPU))
                except Exception:
                    pass
                try:
                    g["power_draw"] = round(pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0, 2)
                except Exception:
                    pass
                try:
                    g["power_limit"] = round(
                        pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0, 2)
                except Exception:
                    pass
                gpus.append(g)
        except Exception:
            pass
        finally:
            try:
                pynvml.nvmlShutdown()
            except Exception:
                pass
        return gpus or None

    def gpu_telemetry(self) -> list:
        via_nvml = self._telemetry_via_pynvml()
        if via_nvml is not None:
            return via_nvml
        rc, out, _ = _run([
            "nvidia-smi",
            "--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,"
            "power.draw,power.limit",
            "--format=csv,noheader,nounits",
        ])
        gpus = []
        if rc != 0:
            return gpus
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 7:
                continue
            def num(x):
                try:
                    return float(x)
                except ValueError:
                    return None
            gpus.append({
                "index": int(parts[0]),
                "util": num(parts[1]),
                "mem_used": num(parts[2]),
                "mem_total": num(parts[3]),
                "temp": num(parts[4]),
                "power_draw": num(parts[5]),
                "power_limit": num(parts[6]),
            })
        return gpus

    def replica_states(self) -> list:
        out = []
        for rid, r in self._replicas.items():
            alive = any(p.poll() is None for p in r["procs"])
            state = "SERVING" if alive else ("FAILED" if r.get("state") != "STOPPED" else "STOPPED")
            out.append({"id": rid, "state": state, "port": r.get("port"),
                        "role": r.get("role")})
        return out

    def mount_states(self) -> list:
        return [{"path": p, "source": m["source"], "ok": os.path.ismount(p)}
                for p, m in self._mounts.items()]

    # -- config / hardware guards ----------------------------------------
    def apply_config(self, cfg: dict) -> dict:
        """Apply live-safe, node-local settings (power cap, persistence)."""
        notes = []
        cap = int(cfg.get("gpu_power_cap_w") or 0)
        if cfg.get("gpu_persistence_mode"):
            _run(["nvidia-smi", "-pm", "1"])
            notes.append("persistence on")
        if cap > 0:
            rc, _, err = _run(["nvidia-smi", "-pl", str(cap)])
            notes.append(f"power cap {cap}W" if rc == 0 else f"power cap failed: {err.strip()}")
        return {"ok": True, "detail": "; ".join(notes) or "no live-safe changes"}

    # -- storage ----------------------------------------------------------
    def mount_storage(self, body: dict) -> dict:
        src = body.get("source", {})
        path = body.get("canonical_path")
        opts = body.get("opts") or ["vers=3", "proto=tcp", "ro", "nofail", "soft"]
        if not path:
            return {"ok": False, "error": "no canonical_path"}
        host = src.get("host")
        export = src.get("export", path)
        requested = "%s:%s" % (host, export)
        if os.path.ismount(path):
            # Never claim a path we didn't mount. Something else (e.g. a legacy
            # kernel NFS entry in fstab) may already own it; reporting the
            # requested source here would make the UI show a share as mounted
            # when it is not.
            actual = self._actual_mount_source(path)
            self._mounts[path] = {"source": actual or "unknown"}
            if actual == requested:
                return {"ok": True, "detail": "already mounted"}
            return {"ok": False,
                    "error": "%s is already mounted from %s (wanted %s) — unmount it first"
                             % (path, actual or "an unknown source", requested)}
        os.makedirs(path, exist_ok=True)
        # modelfsd listens on an advertised ephemeral port, so the NFS client needs
        # both the NFS and MOUNT ports pinned to it.
        port = src.get("port")
        opts = [o for o in opts if not o.startswith(("port=", "mountport="))]
        if port and str(port) != "2049":
            opts += ["port=%s" % port, "mountport=%s" % port]
        cmd = ["mount", "-t", "nfs", "-o", ",".join(opts), f"{host}:{export}", path]
        rc, _, err = _run(["sudo"] + cmd, timeout=30)
        ok = rc == 0 or os.path.ismount(path)
        if ok:
            # record what is ACTUALLY mounted, not what we asked for
            self._mounts[path] = {"source": self._actual_mount_source(path) or requested}
        return {"ok": ok, "detail": "mounted" if ok else "", "error": "" if ok else err.strip()}

    @staticmethod
    def _actual_mount_source(path: str) -> Optional[str]:
        """The device/source currently mounted at `path`, straight from the kernel."""
        try:
            with open("/proc/self/mounts", "r") as fh:
                for line in fh:
                    parts = line.split()
                    if len(parts) >= 2 and parts[1].replace("\\040", " ") == path:
                        return parts[0]
        except Exception:
            pass
        return None

    def unmount_storage(self, path: str) -> dict:
        rc, _, err = _run(["sudo", "umount", path], timeout=30)
        self._mounts.pop(path, None)
        return {"ok": rc == 0, "error": "" if rc == 0 else err.strip()}

    def volumes(self) -> list:
        """Share candidates visible to this node (live free space)."""
        try:
            from ..cluster.detect import detect_volumes
            return detect_volumes()
        except Exception:
            return []

    def share_states(self) -> list:
        """Currently exported shares: [{path, endpoint, ok}]."""
        out = []
        for path, s in self._shares.items():
            alive = s["proc"].poll() is None
            out.append({"path": path, "endpoint": s["endpoint"], "ok": alive})
        return out

    def _free_port(self, bind_ip: str = "") -> int:
        """Let the OS pick a free ephemeral port. The chosen port is advertised to the
        cluster (telemetry -> catalog -> mount), so nothing depends on a fixed number
        and we never clash with the host's kernel nfsd on 2049."""
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        try:
            s.bind((bind_ip or "0.0.0.0", 0))
            return s.getsockname()[1]
        finally:
            s.close()

    def serve_storage(self, body: dict) -> dict:
        """Export one local directory read-only via modelfsd. Multiple shares per node
        are supported — one modelfsd process per share, each on its own port."""
        binary = os.environ.get("MODELFSD_BIN", "modelfsd")
        export = body.get("export_dir")
        if not export:
            return {"ok": False, "error": "export_dir required"}
        # validate: user-selectable paths must actually exist here
        if not os.path.isdir(export):
            return {"ok": False, "error": f"no such directory on this node: {export}"}

        existing = self._shares.get(export)
        if existing and existing["proc"].poll() is None:
            return {"ok": True, "detail": "already shared", "endpoint": existing["endpoint"]}

        listen = body.get("listen") or ""
        if ":" in listen:
            bind_ip, port_s = listen.rsplit(":", 1)
            try:
                port = int(port_s)
            except ValueError:
                port = self._next_share_port()
        else:
            bind_ip = listen or "0.0.0.0"
            port = self._free_port(bind_ip)
        endpoint = "%s:%d" % (bind_ip, port)

        cmd = [binary, "--export", export, "--listen", endpoint]
        allow = body.get("allow") or []
        if allow:
            cmd += ["--allow", ",".join(allow)]
        if body.get("readahead"):
            cmd += ["--readahead", body["readahead"]]
        # Capture stderr so a failed start reports modelfsd's actual complaint
        # (port in use, bad path, bad --allow) rather than a generic message.
        import tempfile
        errf = tempfile.NamedTemporaryFile(prefix="modelfsd-", suffix=".err", delete=False)
        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=errf)
        except FileNotFoundError:
            errf.close()
            os.unlink(errf.name)
            return {"ok": False, "error": f"{binary} not found (build storage/modelfsd)"}
        time.sleep(0.6)
        if proc.poll() is not None:
            errf.flush()
            detail = ""
            try:
                with open(errf.name, "r", errors="replace") as fh:
                    detail = " | ".join(fh.read().strip().splitlines()[-3:])
            except Exception:
                pass
            try:
                os.unlink(errf.name)
            except Exception:
                pass
            return {"ok": False,
                    "error": "modelfsd failed on %s: %s" % (endpoint,
                                                            detail or "exited immediately")}
        self._shares[export] = {"proc": proc, "endpoint": endpoint, "port": port,
                                "errfile": errf.name}
        return {"ok": True, "detail": "serving %s on %s" % (export, endpoint),
                "endpoint": endpoint}

    def unshare_storage(self, body: dict) -> dict:
        """Stop exporting one directory."""
        export = body.get("export_dir")
        s = self._shares.pop(export, None)
        if not s:
            return {"ok": True, "detail": "not shared"}
        if s["proc"].poll() is None:
            s["proc"].terminate()
            time.sleep(1)
            if s["proc"].poll() is None:
                s["proc"].kill()
        if s.get("errfile"):
            try:
                os.unlink(s["errfile"])
            except Exception:
                pass
        return {"ok": True, "detail": "stopped sharing %s" % export}

    # -- model sync -------------------------------------------------------
    def sync_model(self, body: dict) -> dict:
        repo = body.get("repo")
        dest = body.get("dest")
        revision = body.get("revision")
        token = body.get("hf_token")
        if not repo or not dest:
            return {"ok": False, "error": "repo and dest required"}
        env = dict(os.environ)
        if token:
            env["HF_TOKEN"] = token
        code = (
            "from huggingface_hub import snapshot_download;"
            f"snapshot_download({repo!r}, local_dir={dest!r},"
            f"{'revision=' + repr(revision) + ',' if revision else ''} local_dir_use_symlinks=False)"
        )
        try:
            p = subprocess.run(["python3", "-c", code], env=env, capture_output=True,
                               text=True, timeout=body.get("timeout", 3600))
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "download timed out"}
        if p.returncode != 0:
            # never log the token; surface gated/401 distinctly
            tail = (p.stderr or "").strip().splitlines()[-1:] or [""]
            return {"ok": False, "error": tail[0]}
        return {"ok": True, "detail": f"synced {repo} -> {dest}"}

    # -- replica lifecycle (Ray + vLLM) ----------------------------------
    def ensure_replica(self, body: dict) -> dict:
        rid = body["replica_id"]
        if rid in self._replicas and any(p.poll() is None for p in self._replicas[rid]["procs"]):
            return {"ok": True, "state": "SERVING", "detail": "already running"}
        role = body.get("role_in_replica", "head")
        ray = body.get("ray", {})
        env = dict(os.environ)
        env.update({k: str(v) for k, v in (body.get("env") or {}).items()})
        layout = body.get("layout", {})
        executor = layout.get("executor") or ("ray" if layout.get("pp", 1) > 1 else "mp")
        if layout.get("pp_layer_partition"):
            env["VLLM_PP_LAYER_PARTITION"] = layout["pp_layer_partition"]
        procs = []
        head_addr = ray.get("head_addr")
        port = int(ray.get("port", 6379))

        if executor != "ray":
            # single-node, mp executor: no Ray, head runs vLLM across the local GPUs.
            cmd = self._vllm_cmd(body, executor="mp")
            try:
                procs.append(subprocess.Popen(cmd, env=env))
            except FileNotFoundError as exc:
                return {"ok": False, "error": str(exc)}
        elif role == "head":
            _run(["ray", "stop"], timeout=30)
            rc, _, err = _run(["ray", "start", "--head", f"--port={port}"], timeout=60)
            if rc != 0:
                return {"ok": False, "error": f"ray head failed: {err.strip()}"}
            env["RAY_ADDRESS"] = f"{head_addr}:{port}"
            cmd = self._vllm_cmd(body, executor="ray")
            try:
                procs.append(subprocess.Popen(cmd, env=env))
            except FileNotFoundError as exc:
                return {"ok": False, "error": str(exc)}
        else:
            # worker: join the head's Ray cluster (Ray schedules the vLLM workers here)
            rc, _, err = _run(["ray", "start", f"--address={head_addr}:{port}"], timeout=60)
            if rc != 0:
                return {"ok": False, "error": f"ray worker join failed: {err.strip()}"}

        self._replicas[rid] = {"procs": procs, "role": role, "port": body.get("port"),
                               "state": "SERVING", "executor": executor}
        return {"ok": True, "state": "SERVING",
                "detail": f"{role}/{executor} up"
                          + (f" on :{body.get('port')}" if role == "head" else "")}

    def _vllm_cmd(self, body: dict, executor: str = "ray") -> list:
        layout = body.get("layout", {})
        cmd = ["python3", "-m", "vllm.entrypoints.openai.api_server",
               "--model", body["model"],
               "--tensor-parallel-size", str(layout.get("tp", 1)),
               "--pipeline-parallel-size", str(layout.get("pp", 1)),
               "--distributed-executor-backend", executor,
               "--gpu-memory-utilization", str(body.get("gpu_memory_utilization", 0.85)),
               "--dtype", body.get("dtype", "auto"),
               "--host", "0.0.0.0", "--port", str(body.get("port", 8001))]
        if body.get("served_model_name"):
            cmd += ["--served-model-name", body["served_model_name"]]
        if body.get("max_model_len"):
            cmd += ["--max-model-len", str(body["max_model_len"])]
        for a in body.get("vllm_args") or []:
            cmd.append(a)
        return cmd

    def stop_replica(self, replica_id: str, ray: bool = True) -> dict:
        r = self._replicas.get(replica_id)
        if not r:
            return {"ok": True, "detail": "not running here"}
        for p in r["procs"]:
            if p.poll() is None:
                p.terminate()
        time.sleep(3)
        for p in r["procs"]:
            if p.poll() is None:
                p.kill()
        if ray:
            _run(["ray", "stop"], timeout=30)
        r["state"] = "STOPPED"
        return {"ok": True, "detail": "stopped"}

    def shutdown(self):
        for rid in list(self._replicas):
            self.stop_replica(rid, ray=True)
        for path in list(self._shares):
            self.unshare_storage({"export_dir": path})
