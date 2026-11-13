"""Device API (cugen.device / cugen.cuda) and the CPU-node guards.

Runs everywhere: on a GPU node, on a CPU node with cupy installed, and without
cupy. The cupy-blocked checks run in a subprocess with an import hook that makes
``import cupy`` / ``cupyx`` fail, so they also cover a GPU node.
"""
import json
import os
import subprocess
import sys

import numpy as np
import pytest

import cugen

# The subprocess must import the same cugen as this process.
_PKG_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(cugen.__file__)))

BLOCKER = r'''
import sys, importlib.abc
class _Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if name.split(".")[0] in ("cupy", "cupyx", "cupy_backends"):
            raise ImportError(name + " blocked (CPU-only test)")
        return None
sys.meta_path.insert(0, _Block())
'''


def _run(code, block=False, env=None):
    src = (BLOCKER if block else "") + code
    e = dict(os.environ)
    e.pop("CUGEN_DEVICE", None)
    e.pop("CUGEN_MEMORY_LIMIT", None)
    e["PYTHONPATH"] = os.pathsep.join(p for p in (_PKG_ROOT, e.get("PYTHONPATH")) if p)
    if env:
        e.update(env)
    out = subprocess.run([sys.executable, "-c", src], capture_output=True, text=True,
                         env=e, timeout=600)
    assert out.returncode == 0, out.stderr[-3000:]
    return out.stdout.strip().splitlines()[-1]


def _gpu_expected():
    v = os.environ.get("CUGEN_TEST_EXPECT_GPU")
    return None if v is None else v == "1"


def test_import_without_cupy():
    r = json.loads(_run(
        "import json, sys, cugen, cugen._binary_spa, cugen.ld, cugen.impute\n"
        "print(json.dumps({'avail': cugen.cuda.is_available(), 'dev': str(cugen.device()),"
        " 'check': cugen.check_gpu(), 'n': cugen.cuda.device_count(),"
        " 'cupy_loaded': 'cupy' in sys.modules}))", block=True))
    assert r["avail"] is False
    assert r["dev"] == "cpu"
    assert r["n"] == 0
    assert r["cupy_loaded"] is False
    assert r["check"]["available"] is False and "error" in r["check"]
    assert set(r["check"]) == {"available", "error"}


def test_is_available_matches_node():
    exp = _gpu_expected()
    if exp is not None:
        assert cugen.cuda.is_available() is exp
    if not cugen.cuda.is_available():
        assert cugen.cuda.device_count() == 0
        assert str(cugen.device("auto")) == "cpu"
        assert cugen.cuda.unavailable_reason()
        with pytest.raises(RuntimeError):
            cugen.device("cuda")
    else:
        assert cugen.cuda.device_count() >= 1
        assert cugen.cuda.unavailable_reason() is None
        assert str(cugen.device("auto")) == "cuda:0"
        free, total = cugen.cuda.mem_get_info(0)
        assert 0 < free <= total
        assert isinstance(cugen.cuda.get_device_name(0), str)


def test_check_gpu_contract():
    r = cugen.check_gpu()
    if r["available"]:
        assert set(r) == {"available", "device_id", "name", "memory_free_gb", "memory_total_gb"}
    else:
        assert set(r) == {"available", "error"}


def test_device_object_cpu():
    d = cugen.device("cpu")
    assert d.type == "cpu" and d.index is None and d.xp is np
    assert repr(d) == "device(type='cpu')" and str(d) == "cpu"
    a = d.asarray([1.0, 2.0])
    assert d.asnumpy(a).sum() == 3.0
    free, total = d.mem_info()
    assert 0 < free <= total
    assert d.threads >= 1
    d.sync()
    with d as same:                     # context manager is a no-op on CPU
        assert same is d
    assert cugen.device(d) is d
    assert cugen.device("cpu") == d and hash(cugen.device("cpu")) == hash(d)


def test_device_grammar():
    assert str(cugen.device(None)) == str(cugen.device("auto"))
    if cugen.cuda.is_available():
        assert str(cugen.device(0)) == "cuda:0"
        assert str(cugen.device("cuda")) == "cuda:0"
        assert str(cugen.device("CUDA:0")) == "cuda:0"
    else:
        for spec in (0, "cuda", "cuda:0", "gpu"):
            with pytest.raises(RuntimeError):
                cugen.device(spec)


def test_device_env_override():
    assert _run("import cugen; print(cugen.device('auto'))", env={"CUGEN_DEVICE": "cpu"}) == "cpu"


def test_memory_limit_env():
    out = _run("import cugen; print(cugen.device('cpu').mem_info()[1])",
               env={"CUGEN_MEMORY_LIMIT": "3G"})
    assert int(out) == 3 * (1 << 30)


def test_parse_and_format_bytes():
    from cugen._device import format_bytes, parse_bytes
    assert parse_bytes("8G") == 8 << 30
    assert parse_bytes("512MB") == 512 << 20
    assert parse_bytes("1.5 GiB") == int(1.5 * (1 << 30))
    assert parse_bytes(1e9) == 10 ** 9
    assert parse_bytes(None) is None
    assert format_bytes(512) == "512 B"
    assert format_bytes(3 << 30) == "3.0 GB"


def test_bad_device_string():
    with pytest.raises(ValueError):
        cugen.device("tpu")


def test_set_device_cpu_is_noop():
    d = cugen.set_device("cpu")
    assert isinstance(d, cugen.Device) and d.type == "cpu"


def test_resource_error_lists_suggestions():
    e = cugen.CugenResourceError("too big", estimate={"peak_bytes": 1}, suggestions=["a", "b"])
    assert isinstance(e, MemoryError)
    assert e.suggestions == ["a", "b"] and e.estimate == {"peak_bytes": 1}
    assert "- a" in str(e) and "- b" in str(e)


def test_cugen_reader_on_cpu(tmp_path):
    """io.py guard: header/stats/packed reads work whether or not a GPU is present."""
    from cugen.io import CugenReader
    from cugen.write import write_cugen
    g = np.array([[0, 1, 2, 3, 1], [2, 2, 0, 1, 0]], dtype=np.uint8)     # (variants, samples)
    path = str(tmp_path / "chr1.cugen")
    write_cugen(path, g.T.copy(), gidx=np.array([10, 11]))
    r = CugenReader(path)
    assert r.n_samples == 5 and r.n_variants == 2
    assert len(r.read_packed_bytes(0, 2)) == 2 * r.bytes_per_variant
    np.testing.assert_array_equal(r.get_gidx(), [10, 11])


def test_auto_backends_follow_usable_cuda(monkeypatch):
    """With cupy importable but no usable device, every 'auto' resolves to the CPU."""
    import cugen._device as dev_mod
    from cugen import ld
    monkeypatch.setattr(dev_mod, "is_available", lambda: False)
    assert ld._cuda_ok() is False
    assert cugen.device("auto").type == "cpu"
    import importlib
    for name in ("cugen.plot.manhattan", "cugen.plot.qq"):     # modules, not the re-exported functions
        assert importlib.import_module(name)._cuda_usable() is False
