"""Random-access cugen reads: the shared positioned-read helper and the GPU readers.

CPU: ``io._pread_rows`` / ``CugenReader._gather_packed`` against
``read_packed_bytes`` (scattered, contiguous, duplicate rows, gap coalescing,
short reads). GPU: ``read_indices_to_gpu`` (missing="zero"|"nan"), the batched
reader and ``read_indices_raw_to_gpu`` against ``read_to_gpu`` and the known
dosages, on a synthetic file with missing calls. Phased files are refused.
"""
import os

import numpy as np
import pytest

from conftest import requires_gpu, simulate_haplotypes
from cugen import io as cio
from cugen.io import CugenReader


@pytest.fixture
def scattered_cugen(write_cugen_file):
    dos = simulate_haplotypes(37, 120, seed=3, missing_rate=0.05)      # 37 samples -> 10 bytes/variant
    return write_cugen_file(dos, "chr1.cugen"), dos


def _records(r, rows):
    bpv = r.bytes_per_variant
    return np.stack([np.frombuffer(r.read_packed_bytes(int(i), int(i) + 1), np.uint8)
                     for i in rows]).reshape(len(rows), bpv)


@pytest.mark.parametrize("rows", [
    [0], [5, 6, 7, 8], [0, 2, 3, 50, 51, 119], [1, 1, 4, 4, 4, 9], list(range(120)), [],
])
@pytest.mark.parametrize("max_gap", [0, 1, 64])
def test_pread_rows_matches_packed_bytes(scattered_cugen, rows, max_gap):
    path, _ = scattered_cugen
    r = CugenReader(path)
    rows = np.asarray(rows, dtype=np.int64)
    fd = os.open(path, os.O_RDONLY)
    try:
        got = cio._pread_rows(fd, r.data_offset, r.bytes_per_variant, rows, max_gap=max_gap)
    finally:
        os.close(fd)
    assert got.shape == (len(rows), r.bytes_per_variant) and got.dtype == np.uint8
    if len(rows):
        np.testing.assert_array_equal(got, _records(r, rows))


def test_gather_packed_flat_and_identical(scattered_cugen):
    path, _ = scattered_cugen
    r = CugenReader(path)
    rows = np.array([3, 4, 10, 11, 12, 90])
    buf = r._gather_packed(rows)
    assert buf.ndim == 1 and buf.size == len(rows) * r.bytes_per_variant
    np.testing.assert_array_equal(buf.reshape(len(rows), -1), _records(r, rows))


def test_short_reads_are_completed(scattered_cugen, monkeypatch):
    path, _ = scattered_cugen
    r = CugenReader(path)
    real = os.preadv

    def dribble(fd, bufs, off):                     # at most 3 bytes per call
        mv = memoryview(bufs[0])[:3]
        return real(fd, [mv], off)

    monkeypatch.setattr(cio.os, "preadv", dribble)
    rows = np.array([0, 1, 2, 40, 119])
    np.testing.assert_array_equal(r._gather_packed(rows).reshape(len(rows), -1), _records(r, rows))


def test_read_past_end_raises(scattered_cugen):
    path, _ = scattered_cugen
    r = CugenReader(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        with pytest.raises(IOError):
            cio._pread_rows(fd, r.data_offset, r.bytes_per_variant, np.array([r.n_variants + 5]))
    finally:
        os.close(fd)


def test_phased_files_refused(phased_cugen):
    ref, _, _ = phased_cugen
    r = CugenReader(ref)
    for call in (lambda: r.read_indices_raw_to_gpu([0, 1]),
                 lambda: r.read_indices_to_gpu([0, 1], missing="nan"),
                 lambda: r.read_indices_to_gpu_batched([0, 1], missing="nan")):
        with pytest.raises(ValueError, match="PHASED"):
            call()


@requires_gpu
def test_gpu_readers_zero_nan_raw(scattered_cugen):
    import cupy as cp
    path, dos = scattered_cugen                     # dos: (variants, samples), 3 = missing
    r = CugenReader(path)
    idx = np.array([50, 3, 4, 5, 119, 0, 77, 76])   # unsorted, with runs
    truth = dos[idx].T                               # (samples, k)
    raw = cp.asnumpy(r.read_indices_raw_to_gpu(idx))
    np.testing.assert_array_equal(raw, truth)
    zero = cp.asnumpy(r.read_indices_to_gpu(idx))
    np.testing.assert_array_equal(zero, np.where(truth == 3, 0, truth).astype(np.float32))
    nan = cp.asnumpy(r.read_indices_to_gpu(idx, missing="nan"))
    np.testing.assert_array_equal(np.isnan(nan), truth == 3)
    np.testing.assert_array_equal(np.nan_to_num(nan, nan=0.0), zero)
    for miss in ("zero", "nan"):
        b = cp.asnumpy(r.read_indices_to_gpu_batched(idx, missing=miss))
        a = cp.asnumpy(r.read_indices_to_gpu(idx, missing=miss))
        np.testing.assert_array_equal(np.isnan(a), np.isnan(b))
        np.testing.assert_array_equal(np.nan_to_num(a), np.nan_to_num(b))
    seq = cp.asnumpy(r.read_to_gpu(0, r.n_variants))[:, idx]
    np.testing.assert_array_equal(seq, zero)
    with pytest.raises(ValueError):
        r.read_indices_to_gpu(idx, missing="drop")
    assert r.read_indices_raw_to_gpu([]).shape == (r.n_samples, 0)
