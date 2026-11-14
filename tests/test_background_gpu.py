"""cugen.background: device="cuda" equals device="cpu" (same code path through xp)."""
import numpy as np
import pytest

import cugen as cg
from conftest import make_cohort_files
from cugen import background as bg

pytestmark = [pytest.mark.gpu,
              pytest.mark.skipif(not cg.cuda.is_available(), reason="no CUDA device")]


@pytest.fixture(scope="module")
def data(tmp_path_factory):
    return make_cohort_files(tmp_path_factory.mktemp("bg_gpu"), n=260, m_per_chr=(80, 60, 50), seed=5)


def test_decode_gpu_equals_cpu(data):
    for g in (data["bed"], data["cg"]):
        np.testing.assert_array_equal(g.read_counts(dev="cuda"), g.read_counts())


def test_cugenie_gpu_equals_cpu(data):
    kw = dict(block_size=30, loco="both", verbose=False)
    a = cg.fit_cugenie(data["bed"], data["y"], data["C"], device="cpu", **kw)
    b = cg.fit_cugenie(data["bed"], data["y"], data["C"], device="cuda", **kw)
    for v in ("masked", "refit"):
        np.testing.assert_allclose(b[v].predictions, a[v].predictions, atol=1e-9)


def test_cugenie_k_gpu_equals_cpu(data):
    a = cg.fit_cugenie_k(data["bed"], data["y"], data["C"], device="cpu", offset="both", verbose=False)
    b = cg.fit_cugenie_k(data["bed"], data["y"], data["C"], device="cuda", offset="both", verbose=False)
    for v in ("loo", "insample"):
        np.testing.assert_allclose(b[v].offsets, a[v].offsets, atol=1e-8)


def test_ultraloco_gpu_equals_cpu(data):
    a = cg.fit_ultraloco(data["cg"], data["y"], data["C"], device="cpu", verbose=False).gwas()
    b = cg.fit_ultraloco(data["cg"], data["y"], data["C"], device="cuda", verbose=False).gwas()
    np.testing.assert_allclose(b.Z.to_numpy(), a.Z.to_numpy(), atol=1e-4)


def test_gwas_blocked_gpu_equals_cpu(data):
    L = cg.fit_cugenie(data["bed"], data["y"], data["C"], block_size=30, verbose=False, device="cpu")
    for g in (data["bed"], data["cg"]):
        a = bg.gwas_blocked(g, L, device="cpu", verbose=False, check_samples=False)
        b = bg.gwas_blocked(g, L, device="cuda", verbose=False, check_samples=False)
        np.testing.assert_allclose(b.Z.to_numpy(), a.Z.to_numpy(), atol=1e-5)


def test_gpu_resource_error_is_raised_before_allocating(data):
    with pytest.raises(cg.CugenResourceError):
        cg.fit_ultraloco(data["cg"], data["y"], data["C"], device="cuda", memory_budget="1MB", verbose=False)


@pytest.mark.skipif(cg.cuda.device_count() < 2, reason="needs at least 2 GPUs")
def test_second_device_keeps_everything_on_it(data):
    import cupy as cp
    a = cg.fit_cugenie(data["bed"], data["y"], data["C"], block_size=30, verbose=False, device="cpu")
    with cp.cuda.Device(0):
        before = cp.get_default_memory_pool().used_bytes()
    b = cg.fit_cugenie(data["bed"], data["y"], data["C"], block_size=30, verbose=False, device="cuda:1")
    np.testing.assert_allclose(b.predictions, a.predictions, atol=1e-9)
    k = cg.fit_cugenie_k(data["bed"], data["y"], data["C"], verbose=False, device="cuda:1")
    assert np.isfinite(k.offsets).all()
    with cp.cuda.Device(0):
        assert cp.get_default_memory_pool().used_bytes() == before
