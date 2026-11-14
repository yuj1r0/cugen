"""``cugen.cuda``: PyTorch-style CUDA queries (``cugen.cuda.is_available()`` etc.).

All functions are safe to call without cupy or without a GPU; results are cached."""
from ._device import device_count, get_device_name, is_available, mem_get_info, unavailable_reason

__all__ = ["is_available", "device_count", "get_device_name", "mem_get_info", "unavailable_reason"]
