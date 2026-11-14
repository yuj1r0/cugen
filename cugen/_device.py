"""PyTorch-style device selection for CuGen (CPU or CUDA), plus memory queries.

One device grammar is used library-wide (``DeviceLike``)::

    int N | "auto" | "cpu" | "cuda" | "cuda:N" | Device

An int keeps its historical meaning, the CUDA device index, so existing ``device=0``
arguments are unchanged. Runtime decisions use :func:`is_available` (cupy imports AND a
device exists AND a 1-element op runs), never the bare fact that ``import cupy`` worked:
on a CPU node with cupy installed the import succeeds but every CUDA call fails.

    import cugen as cg
    cg.cuda.is_available()            # cupy imports AND a device exists AND a 1-element op runs
    dev = cg.device("auto")           # CUGEN_DEVICE env var, else "cuda" if available, else "cpu"
    dev = cg.device("cuda:1")         # explicit
    xp = dev.xp                       # numpy or cupy
    free, total = dev.mem_info()      # bytes; CPU honours the job's cgroup / rlimit memory limit

Nothing here needs psutil: CPU memory comes from /proc/meminfo, the process's own cgroup
memory limit (cgroup v1 or v2, as Slurm sets it) and RLIMIT_AS. ``CUGEN_MEMORY_LIMIT``
(e.g. ``8G``) overrides the detected limit.
"""
from __future__ import annotations

import os
import threading
from typing import Union

import numpy as np

__all__ = ["Device", "DeviceLike", "device", "CugenResourceError", "is_available", "device_count",
           "get_device_name", "mem_get_info", "cpu_threads", "parse_bytes", "format_bytes"]


class CugenResourceError(MemoryError):
    """Raised before allocating when a computation will not fit the device's memory.

    ``estimate`` holds the resource estimate dict, ``suggestions`` a list of options
    that would fit (or are the next things to try)."""

    def __init__(self, message, estimate=None, suggestions=None):
        self.estimate = estimate or {}
        self.suggestions = list(suggestions or [])
        if self.suggestions:
            message = message + "\n  Options that reduce memory:\n" + "\n".join(
                f"    - {s}" for s in self.suggestions)
        super().__init__(message)


# ----------------------------------------------------------------------------- CUDA probing
_LOCK = threading.Lock()
_CUDA = {"checked": False, "ok": False, "count": 0, "error": None}


def _probe_cuda():
    with _LOCK:
        if _CUDA["checked"]:
            return _CUDA
        _CUDA["checked"] = True
        try:
            import cupy as cp
        except Exception as e:  # noqa: BLE001 - ImportError or a broken install
            _CUDA["error"] = str(e)
            return _CUDA
        try:
            cnt = int(cp.cuda.runtime.getDeviceCount())
            if cnt <= 0:
                _CUDA["error"] = "no CUDA device"
                return _CUDA
            a = cp.ones(1, dtype=cp.float32)
            float((a + a).sum())          # forces a kernel launch + device->host copy
            _CUDA.update(ok=True, count=cnt)
        except Exception as e:  # noqa: BLE001 - driver/runtime errors are not ImportError
            _CUDA["error"] = str(e)
        return _CUDA


def is_available() -> bool:
    """True iff cupy imports, ``getDeviceCount() > 0`` and a 1-element op succeeds (cached)."""
    return bool(_probe_cuda()["ok"])


def unavailable_reason():
    """Why CUDA is not available (None when it is)."""
    s = _probe_cuda()
    return None if s["ok"] else s["error"]


def device_count() -> int:
    return int(_probe_cuda()["count"])


def get_device_name(i: int = 0) -> str:
    if not is_available():
        raise RuntimeError(f"CUDA is not available: {unavailable_reason()}")
    import cupy as cp
    name = cp.cuda.runtime.getDeviceProperties(int(i))["name"]
    return name.decode() if isinstance(name, bytes) else str(name)


def mem_get_info(i: int = 0):
    """(free, total) bytes of CUDA device i (like torch.cuda.mem_get_info)."""
    if not is_available():
        raise RuntimeError(f"CUDA is not available: {unavailable_reason()}")
    import cupy as cp
    with cp.cuda.Device(int(i)):
        free, total = cp.cuda.runtime.memGetInfo()
    return int(free), int(total)


# ----------------------------------------------------------------------------- CPU memory
_UNITS = {"": 1, "B": 1, "K": 1 << 10, "KB": 1 << 10, "KIB": 1 << 10, "M": 1 << 20, "MB": 1 << 20,
          "MIB": 1 << 20, "G": 1 << 30, "GB": 1 << 30, "GIB": 1 << 30, "T": 1 << 40, "TB": 1 << 40,
          "TIB": 1 << 40}


def parse_bytes(v) -> int:
    """'8G', '512MB', 1e9 -> bytes."""
    if v is None:
        return None
    if isinstance(v, (int, float, np.integer, np.floating)):
        return int(v)
    s = str(v).strip().upper().replace(" ", "")
    num = s.rstrip("KMGTIB")
    unit = s[len(num):]
    return int(float(num) * _UNITS[unit])


def format_bytes(b) -> str:
    b = float(b)
    for u in ("B", "KB", "MB", "GB", "TB"):
        if abs(b) < 1024 or u == "TB":
            return f"{b:.1f} {u}" if u != "B" else f"{int(b)} B"
        b /= 1024.0
    return f"{b:.1f} TB"


def _read_int(path):
    try:
        with open(path) as fh:
            s = fh.read().strip()
        if s == "max":
            return None
        v = int(s)
        return v if v < (1 << 60) else None
    except (OSError, ValueError):
        return None


def _stat(base, fname):
    try:
        with open(os.path.join(base, fname)) as fh:
            return dict((l.split()[0], int(l.split()[1])) for l in fh if len(l.split()) == 2)
    except (OSError, ValueError):
        return None


def _v1_usage(base):
    """Non-reclaimable usage of a v1 memory cgroup: anonymous + shmem + mapped file pages.
    (memory.usage_in_bytes also counts the page cache left by earlier reads, which the kernel
    reclaims under pressure; counting it made 'free' far too pessimistic.)"""
    st = _stat(base, "memory.stat")
    if st and "total_rss" in st:
        return int(st.get("total_rss", 0) + st.get("total_shmem", 0) + st.get("total_mapped_file", 0))
    return _read_int(os.path.join(base, "memory.usage_in_bytes"))


def _v2_usage(base):
    st = _stat(base, "memory.stat")
    if st and "anon" in st:
        return int(st.get("anon", 0) + st.get("shmem", 0) + st.get("file_mapped", 0))
    return _read_int(os.path.join(base, "memory.current"))


def _cgroup_memory():
    """(limit, usage) of the tightest memory cgroup enclosing this process, or (None, None).

    Walks up from the process's own cgroup (Slurm puts the job limit on an ancestor of
    the step/task cgroup); cgroup v1 and v2."""
    try:
        with open("/proc/self/cgroup") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return None, None
    best = (None, None)
    for ln in lines:
        parts = ln.split(":", 2)
        if len(parts) != 3:
            continue
        _, ctrl, path = parts
        if ctrl == "":
            roots, limf, v2 = ("/sys/fs/cgroup",), "memory.max", True
        elif "memory" in ctrl.split(","):
            roots, limf, v2 = ("/sys/fs/cgroup/memory", "/sys/fs/cgroup/memory,cpuacct"), \
                "memory.limit_in_bytes", False
        else:
            continue
        for root in roots:
            if not os.path.isdir(root):
                continue
            p = path.rstrip("/")
            while True:
                base = root + p
                lim = _read_int(os.path.join(base, limf))
                if lim is not None and (best[0] is None or lim < best[0]):
                    use = _v2_usage(base) if v2 else _v1_usage(base)
                    best = (lim, use)
                if p in ("", "/"):
                    break
                p = p.rsplit("/", 1)[0]
            break
    return best


def _meminfo():
    out = {}
    try:
        with open("/proc/meminfo") as fh:
            for ln in fh:
                k, v = ln.split(":", 1)
                out[k] = int(v.split()[0]) * 1024
    except (OSError, ValueError):
        pass
    return out


def _rss_bytes():
    try:
        with open("/proc/self/status") as fh:
            for ln in fh:
                if ln.startswith("VmRSS:"):
                    return int(ln.split()[1]) * 1024
    except (OSError, ValueError):
        pass
    return 0


def cpu_mem_info():
    """(free, total) bytes usable by this process on the CPU side."""
    mi = _meminfo()
    total = mi.get("MemTotal") or (os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    free = mi.get("MemAvailable", mi.get("MemFree", total))
    env = os.environ.get("CUGEN_MEMORY_LIMIT")
    if env:
        lim = parse_bytes(env)
        return max(lim - _rss_bytes(), 0), lim
    lim, use = _cgroup_memory()
    if lim is not None and lim < total:
        total = lim
        free = min(free, max(lim - (use if use is not None else _rss_bytes()), 0))
    try:
        import resource
        soft, _ = resource.getrlimit(resource.RLIMIT_AS)
        if soft not in (resource.RLIM_INFINITY, -1) and soft < total:
            total = soft
            free = min(free, max(soft - _rss_bytes(), 0))
    except (ImportError, ValueError, OSError):
        pass
    return int(free), int(total)


def cpu_threads() -> int:
    for v in ("OMP_NUM_THREADS", "SLURM_CPUS_PER_TASK"):
        try:
            n = int(os.environ.get(v, ""))
            if n > 0:
                return n
        except ValueError:
            pass
    try:
        return len(os.sched_getaffinity(0))
    except (AttributeError, OSError):
        return os.cpu_count() or 1


# ----------------------------------------------------------------------------- Device
class Device:
    """A compute device: ``type`` 'cpu' or 'cuda', ``index``, array module ``xp``."""

    def __init__(self, type: str, index: int | None = None):
        if type not in ("cpu", "cuda"):
            raise ValueError(f"device type must be 'cpu' or 'cuda', got {type!r}")
        self.type = type
        self.index = (index or 0) if type == "cuda" else None
        if type == "cuda":
            if not is_available():
                raise RuntimeError(f"CUDA device requested but CUDA is not available: {unavailable_reason()}")
            if self.index >= device_count():
                raise RuntimeError(f"cuda:{self.index} requested but only {device_count()} device(s)")
            import cupy as cp
            self.xp = cp
        else:
            self.xp = np

    # -- array helpers
    def asarray(self, a, dtype=None):
        if self.type == "cuda":
            with self:
                return self.xp.asarray(a, dtype=dtype)
        return np.asarray(a, dtype=dtype)

    def asnumpy(self, a):
        if self.type == "cuda" and not isinstance(a, np.ndarray) and hasattr(a, "get"):
            return a.get()
        return np.asarray(a)

    def sync(self):
        if self.type == "cuda":
            self.xp.cuda.Device(self.index).synchronize()

    def mem_info(self):
        """(free, total) bytes. CUDA: device memory (+ cupy pool's free blocks); CPU: see cpu_mem_info."""
        if self.type == "cuda":
            free, total = mem_get_info(self.index)
            try:
                pool = self.xp.get_default_memory_pool()
                free += pool.total_bytes() - pool.used_bytes()
            except Exception:  # noqa: BLE001
                pass
            return int(free), int(total)
        return cpu_mem_info()

    @property
    def free_memory(self) -> int:
        return self.mem_info()[0]

    @property
    def name(self) -> str:
        if self.type == "cuda":
            return get_device_name(self.index)
        try:
            with open("/proc/cpuinfo") as fh:
                for ln in fh:
                    if ln.startswith("model name"):
                        return ln.split(":", 1)[1].strip()
        except OSError:
            pass
        return "cpu"

    @property
    def threads(self) -> int:
        return cpu_threads()

    def free_pool(self):
        if self.type == "cuda":
            self.xp.get_default_memory_pool().free_all_blocks()

    def describe(self) -> str:
        free, total = self.mem_info()
        if self.type == "cuda":
            return (f"device cuda:{self.index} ({self.name}), free {format_bytes(free)} of "
                    f"{format_bytes(total)}")
        return (f"device cpu ({self.threads} threads), free RAM {format_bytes(free)} of "
                f"{format_bytes(total)}")

    # -- context manager (selects the CUDA device; no-op on CPU)
    def __enter__(self):
        if self.type == "cuda":
            self._ctx = self.xp.cuda.Device(self.index)
            self._ctx.__enter__()
        return self

    def __exit__(self, *exc):
        if self.type == "cuda":
            self._ctx.__exit__(*exc)
        return False

    def __eq__(self, other):
        return isinstance(other, Device) and (self.type, self.index) == (other.type, other.index)

    def __hash__(self):
        return hash((self.type, self.index))

    def __str__(self):
        return self.type if self.type == "cpu" else f"cuda:{self.index}"

    def __repr__(self):
        return (f"device(type='{self.type}')" if self.type == "cpu"
                else f"device(type='cuda', index={self.index})")


DeviceLike = Union[int, str, Device, None]


def device(spec: DeviceLike = "auto") -> Device:
    """``"auto"`` | ``"cpu"`` | ``"cuda"`` | ``"cuda:N"`` | int N | a Device -> Device.

    ``"auto"`` (or None) honours ``CUGEN_DEVICE`` (same grammar), else CUDA device 0 if
    :func:`is_available`, else CPU. An int N means CUDA device N."""
    if isinstance(spec, Device):
        return spec
    if spec is None:
        spec = "auto"
    if isinstance(spec, (int, np.integer)):
        return Device("cuda", int(spec))
    s = str(spec).strip().lower()
    if s == "auto":
        env = os.environ.get("CUGEN_DEVICE", "").strip().lower()
        if env and env != "auto":
            return device(env)
        return Device("cuda", 0) if is_available() else Device("cpu")
    if s == "cpu":
        return Device("cpu")
    if s == "cuda" or s == "gpu":
        return Device("cuda", 0)
    if s.startswith("cuda:"):
        return Device("cuda", int(s.split(":", 1)[1]))
    raise ValueError(f"unknown device {spec!r} (use 'auto', 'cpu', 'cuda' or 'cuda:N')")
