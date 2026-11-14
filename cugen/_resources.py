"""Resource estimates (``estimate_only=True``) and the pre-allocation fit check.

Heavy public functions take ``estimate_only=False`` and ``memory_budget=None`` (bytes or a
string such as ``"8G"``). The estimate is a dict with ``method``, ``device``, ``peak_bytes``,
``work_bytes``, ``budget_bytes``, ``fits``, ``flops``, ``flops_by_stage``, ``dominant`` and
``suggestions``; a call that will not fit raises :class:`cugen.CugenResourceError` before it
allocates anything.
"""
from __future__ import annotations

from ._device import CugenResourceError, format_bytes

DEFAULT_FRACTION = 0.85     # share of the free device memory a call may plan to use
BASELINE_BYTES = 200 << 20  # interpreter + numpy/scipy/pandas/pyarrow, measured ~0.2 GB


def budget(dev, memory_budget=None):
    """Bytes this call may newly allocate: memory_budget (bytes or '6G') or 85 % of the free memory
    (CPU: cgroup/rlimit limit minus non-reclaimable usage; GPU: free VRAM)."""
    from ._device import parse_bytes
    if memory_budget is not None:
        return parse_bytes(memory_budget)
    return int(dev.free_memory * DEFAULT_FRACTION)


def finish(est, dev, memory_budget, suggestions, what, geno=None):
    """Attach budget info (host-side reader metadata + interpreter baseline are added to the
    peak on CPU, where they share the budget)."""
    b = budget(dev, memory_budget)
    est["work_bytes"] = int(est["peak_bytes"])           # what this call allocates on the device
    if dev.type == "cpu":
        # whole-process peak (comparable with /usr/bin/time) = what the process holds now (interpreter,
        # reader metadata, allocator-retained memory, user data) + this call's allocations. A reader's
        # resident size is typically ~2x its array bytes, so a modelled baseline alone underestimates.
        # The fit check uses work_bytes only: the free memory already excludes what is resident now.
        from ._device import _rss_bytes
        now = max(int(_rss_bytes()), (int(geno.resident_bytes()) if geno is not None else 0) + BASELINE_BYTES)
        est["peak_breakdown"] = dict(est.get("peak_breakdown", {}), process_resident_now=now)
        est["peak_bytes"] = int(est["peak_bytes"] + now)
    est["device"] = str(dev)
    est["budget_bytes"] = int(b)
    est["fits"] = bool(est["work_bytes"] <= b)
    est["peak"] = format_bytes(est["peak_bytes"])
    est["budget"] = format_bytes(b)
    est["suggestions"] = list(suggestions)
    return est


def check(est, what):
    if not est["fits"]:
        raise CugenResourceError(
            f"{what}: estimated memory {format_bytes(est['work_bytes'])} exceeds the budget {est['budget']} on "
            f"{est['device']} (dominant: {est.get('dominant', '?')})", estimate=est,
            suggestions=est.get("suggestions"))


def flops_str(f):
    return f"{f / 1e12:.2f} TFLOP" if f >= 1e11 else f"{f / 1e9:.2f} GFLOP"
