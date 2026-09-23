"""Node hardware detection for the vllm-multi-gpu control plane.

Python port of the ``det_*`` hardware-detection functions in the repo's
``setup.sh``. The node agent runs this at ``register`` time so a fresh worker
can report its own inventory (GPUs, NVLink, IOMMU, RDMA HCA/port/GID, RoCE NICs)
and the derived NCCL fabric tuning — the same logic setup.sh uses to seed a
``.env``, but as structured data for the control plane (see docs/02 and docs/07).

Design goals:
  * Every probe is best-effort. Missing tools/paths yield empty/None, never an
    exception. ``detect_node()`` NEVER raises, even on a box with no GPUs/RDMA.
  * Standard library only. Targets Python 3.9+ (dev host 3.9, container 3.12).
  * Derivation rules mirror setup.sh EXACTLY:
      - NCCL_IB_HCA pins ``:port`` only when the card exposes >1 port.
      - NCCL_P2P_DISABLE / DISABLE_CUSTOM_ALL_REDUCE are on ONLY when NVLink is
        absent AND IOMMU is enabled (broken IOMMU-translated PCIe P2P).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# low-level helpers
# ---------------------------------------------------------------------------

_IB_ROOT = "/sys/class/infiniband"
_IOMMU_GROUPS = "/sys/kernel/iommu_groups"
_ZERO_GID = "0000:0000:0000:0000:0000:0000:0000:0000"


def _run(cmd: List[str]) -> str:
    """Run a command, returning its stdout (stripped). Empty string on any
    failure (missing binary, non-zero exit, timeout)."""
    try:
        out = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            timeout=15,
            check=False,
        )
        return out.stdout.decode("utf-8", "replace").strip()
    except Exception:
        return ""


def _read_file(path: str) -> str:
    """Read a sysfs/text file, returning its stripped contents or ''."""
    try:
        with open(path, "r") as fh:
            return fh.read().strip()
    except Exception:
        return ""


def _listdir(path: str) -> List[str]:
    """List a directory, returning [] on any error."""
    try:
        return os.listdir(path)
    except Exception:
        return []


def _sorted_numeric(names: List[str]) -> List[str]:
    """Sort entries numerically when they are all integers (mirrors ``sort -n``),
    otherwise fall back to a plain lexical sort."""
    try:
        return sorted(names, key=lambda n: int(n))
    except (ValueError, TypeError):
        return sorted(names)


# ---------------------------------------------------------------------------
# GPU detection  (mirrors det_gpu_count, plus richer per-GPU inventory)
# ---------------------------------------------------------------------------

def detect_gpus() -> List[Dict[str, object]]:
    """Return per-GPU inventory via nvidia-smi. [] when no GPUs / no driver."""
    out = _run([
        "nvidia-smi",
        "--query-gpu=index,name,memory.total,uuid",
        "--format=csv,noheader,nounits",
    ])
    gpus: List[Dict[str, object]] = []
    if not out:
        return gpus
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 4:
            continue
        idx_s, name, mem_s, uuid = parts[0], parts[1], parts[2], parts[3]
        try:
            index = int(idx_s)
        except ValueError:
            continue
        try:
            mem_total_mb = int(float(mem_s))
        except ValueError:
            mem_total_mb = None
        gpus.append(
            {
                "index": index,
                "name": name,
                "mem_total_mb": mem_total_mb,
                "uuid": uuid,
            }
        )
    return gpus


def detect_nvlink_active() -> bool:
    """True only if at least one NVLink is active (mirrors det_nvlink):
    an output line matches ``Link N:`` AND the output does not mention
    "inactive" anywhere."""
    out = _run(["nvidia-smi", "nvlink", "--status"])
    if not out:
        return False
    has_link = re.search(r"Link [0-9]+:", out) is not None
    inactive = "inactive" in out.lower()
    return has_link and not inactive


def detect_iommu() -> bool:
    """True if /sys/kernel/iommu_groups is non-empty (mirrors det_iommu)."""
    return len(_listdir(_IOMMU_GROUPS)) > 0


# ---------------------------------------------------------------------------
# RDMA detection  (mirrors det_hca/det_ports/det_gid/det_netdev/det_ip)
# ---------------------------------------------------------------------------

def detect_hcas() -> List[str]:
    """All InfiniBand/RoCE HCAs under /sys/class/infiniband (sorted)."""
    return sorted(_listdir(_IB_ROOT))


def detect_ports(hca: str) -> List[int]:
    """Numerically-sorted port numbers for an HCA (mirrors det_ports)."""
    names = _listdir(os.path.join(_IB_ROOT, hca, "ports"))
    ports: List[int] = []
    for n in names:
        try:
            ports.append(int(n))
        except ValueError:
            continue
    return sorted(ports)


def detect_gid_index(hca: str, port: int) -> Optional[int]:
    """RoCEv2 IPv4 GID index for ``hca`` port ``port`` (mirrors det_gid).

    Scans GID indices in numeric order; returns the first whose type contains
    "v2" and whose GID is an IPv4-mapped address (contains ``ffff:``) and is not
    the all-zero GID. None if none found."""
    base = os.path.join(_IB_ROOT, hca, "ports", str(port))
    gid_names = _sorted_numeric(_listdir(os.path.join(base, "gids")))
    for i in gid_names:
        typ = _read_file(os.path.join(base, "gid_attrs", "types", i)).lower()
        gid = _read_file(os.path.join(base, "gids", i)).lower()
        if "v2" not in typ:
            continue
        # IPv4-mapped GID (::ffff:a.b.c.d) — bash matched "*ffff:*".
        if "ffff:" in gid and gid != _ZERO_GID:
            try:
                return int(i)
            except ValueError:
                return None
    return None


def _rdma_link_netdevs() -> Dict[str, str]:
    """Parse ``rdma link show`` into {"<hca>/<port>": "<netdev>"} (mirrors the
    det_netdev sed). Empty when the rdma tool is unavailable."""
    out = _run(["rdma", "link", "show"])
    mapping: Dict[str, str] = {}
    if not out:
        return mapping
    for line in out.splitlines():
        m_dev = re.search(r"link\s+(\S+/\d+)\s", line)
        m_net = re.search(r"netdev\s+(\S+)", line)
        if m_dev and m_net:
            mapping[m_dev.group(1)] = m_net.group(1)
    return mapping


def detect_netdev(hca: str, port: int, cache: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Netdev bound to ``hca/port`` via ``rdma link show`` (mirrors det_netdev)."""
    mapping = cache if cache is not None else _rdma_link_netdevs()
    return mapping.get("{}/{}".format(hca, port))


def detect_ip(netdev: str) -> Optional[str]:
    """First IPv4 address on ``netdev`` (mirrors det_ip)."""
    if not netdev:
        return None
    out = _run(["ip", "-o", "-4", "addr", "show", "dev", netdev])
    for line in out.splitlines():
        fields = line.split()
        # `ip -o -4` fields: <idx>: <dev> inet <cidr> ...  → cidr is field[3]
        for tok in fields:
            if "/" in tok and re.match(r"^\d+\.\d+\.\d+\.\d+/\d+$", tok):
                return tok.split("/")[0]
    return None


def detect_rdma() -> List[Dict[str, object]]:
    """Structured RDMA inventory: one entry per HCA with its ports, the RoCEv2
    IPv4 GID index / netdev / fabric IP of its first port (mirrors setup.sh,
    which keys the fabric tuning off the first port)."""
    link_cache = _rdma_link_netdevs()
    rdma: List[Dict[str, object]] = []
    for hca in detect_hcas():
        ports = detect_ports(hca)
        first_port = ports[0] if ports else None
        gid_index: Optional[int] = None
        netdev: Optional[str] = None
        fabric_ip: Optional[str] = None
        if first_port is not None:
            gid_index = detect_gid_index(hca, first_port)
            netdev = detect_netdev(hca, first_port, cache=link_cache)
            if netdev:
                fabric_ip = detect_ip(netdev)
        rdma.append(
            {
                "hca": hca,
                "ports": ports,
                "gid_index": gid_index,
                "netdev": netdev,
                "fabric_ip": fabric_ip,
            }
        )
    return rdma


def detect_mgmt_ip() -> Optional[str]:
    """First non-loopback IPv4 on the box (mirrors det_mgmt_ip)."""
    out = _run(["ip", "-o", "-4", "addr", "show"])
    for line in out.splitlines():
        fields = line.split()
        # Skip the loopback interface (field index 1 is the dev name).
        if len(fields) >= 2 and fields[1] == "lo":
            continue
        for tok in fields:
            if "/" in tok and re.match(r"^\d+\.\d+\.\d+\.\d+/\d+$", tok):
                ip = tok.split("/")[0]
                if not ip.startswith("127."):
                    return ip
    return None


# ---------------------------------------------------------------------------
# derived defaults  (mirrors the "Derived, non-hardcoded defaults" block)
# ---------------------------------------------------------------------------

def derive(
    nvlink_active: bool,
    iommu: bool,
    rdma: List[Dict[str, object]],
) -> Dict[str, object]:
    """Compute the derived NCCL/fabric defaults from detected inventory,
    following setup.sh's rules exactly."""
    primary = rdma[0] if rdma else None

    # NCCL_IB_HCA: pin ":<first_port>" ONLY when the card exposes >1 port.
    nccl_ib_hca: Optional[str] = None
    if primary is not None:
        hca = primary.get("hca")
        ports = primary.get("ports") or []
        if hca:
            if isinstance(ports, list) and len(ports) > 1:
                nccl_ib_hca = "{}:{}".format(hca, ports[0])
            else:
                nccl_ib_hca = str(hca)

    # Broken IOMMU-translated PCIe P2P: disable P2P + custom all-reduce ONLY
    # when NVLink is absent AND IOMMU is enabled.
    broken_p2p = (not nvlink_active) and iommu
    nccl_p2p_disable = broken_p2p
    disable_custom_all_reduce = broken_p2p

    # Primary RoCE netdev (local-first) drives the socket ifnames.
    primary_netdev: Optional[str] = primary.get("netdev") if primary else None

    return {
        "nccl_ib_hca": nccl_ib_hca,
        "nccl_p2p_disable": nccl_p2p_disable,
        "disable_custom_all_reduce": disable_custom_all_reduce,
        "nccl_socket_ifname": primary_netdev,
        "gloo_socket_ifname": primary_netdev,
    }


# ---------------------------------------------------------------------------
# public entry point
# ---------------------------------------------------------------------------

def detect_node() -> dict:
    """Full node inventory. Never raises; unpopulated fields are empty/None."""
    gpus = detect_gpus()
    nvlink_active = detect_nvlink_active()
    iommu = detect_iommu()
    rdma = detect_rdma()
    mgmt_ip = detect_mgmt_ip()
    derived = derive(nvlink_active, iommu, rdma)

    return {
        "gpus": gpus,
        "nvlink_active": nvlink_active,
        "iommu": iommu,
        "rdma": rdma,
        "mgmt_ip": mgmt_ip,
        "derived": derived,
    }


if __name__ == "__main__":
    print(json.dumps(detect_node(), indent=2))
