"""cugen.subset — GPU sample subsetting of .cugen files.

Self-contained port of ``cugen_sample_subset_gpu.py``. Subsetting a cugen is
not just a row-slice: the per-variant ``mu_x / sxx / maf`` statistics in the
header are sample-specific and MUST be recomputed for the new sample set, or
downstream LASSO silently fits zero SNPs (cf. the CRITICAL lesson in
MEMORY.md). This module therefore extracts the requested samples AND
recomputes stats AND repacks 2-bit genotypes, all on the GPU in chunks.

Public API
----------
``subset_cugen_file(input_path, output_path, sample_indices, **kw)``
    Single-file subset. Returns wall-time in seconds.

``subset_cugen_dir(source_dir, output_dir, sample_indices, **kw)``
    Batch chr1..22 subset. Returns dict with per-chr wall times.

``filter_cols(...)``  (PLINK ``keep`` semantics — sample subset)
    Alias for ``subset_cugen_file`` / ``subset_cugen_dir`` (dispatches on
    whether ``input_path`` is a file or directory).

``filter_rows(...)``  (PLINK ``extract`` semantics — variant subset)
    Stub for v0.2 — variant subset of a cugen still raises NotImplementedError.

``sample_indices`` accepted forms
---------------------------------
* ``numpy.ndarray`` of int64 positions in the source cugen
* path to a ``.npy`` file containing the above
* ``(npz_path, key)`` tuple — loads ``key`` from the cohort NPZ (e.g. the
  ``subset_rows_in_base_cugen`` array emitted by :func:`prepare_cohort`)
"""

import multiprocessing as mp
import os
import struct
import time
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from .io import (
    CUGEN_MAGIC,
    CUGEN_VERSION,
    ENCODING_2BIT,
    FLAG_HAS_GIDX_MAP,
    FLAG_HAS_MISSING,
    HEADER_SIZE,
)

from ._stubs import _stub


SampleIndices = Union[
    np.ndarray,
    Sequence[int],
    str,
    Path,
    Tuple[Union[str, Path], str],
]


def _resolve_sample_indices(sample_indices: SampleIndices) -> np.ndarray:
    """Accept ndarray | sequence | .npy path | (npz_path, key)."""
    if isinstance(sample_indices, tuple) and len(sample_indices) == 2:
        npz_path, key = sample_indices
        with np.load(npz_path, allow_pickle=True) as z:
            arr = np.asarray(z[key])
        return arr.astype(np.int64, copy=False)
    if isinstance(sample_indices, (str, Path)):
        return np.load(str(sample_indices)).astype(np.int64, copy=False)
    return np.asarray(sample_indices, dtype=np.int64)


def subset_cugen_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path],
    sample_indices: SampleIndices,
    *,
    chunk_size: int = 8192,
    verbose: bool = True,
    use_pinned: Optional[bool] = None,
    n_buffers: int = 2,
) -> float:
    """GPU-accelerated cugen sample subsetting (single file).

    Extracts the requested samples, recomputes per-variant ``mu_x / sxx /
    maf`` stats, and repacks 2-bit genotypes — all on the GPU in
    ``chunk_size`` blocks. The output cugen has identical layout to the
    source (header → stats → gidx → packed data) but with ``n_samples =
    len(sample_indices)`` and freshly computed stats.

    I/O path: when ``use_pinned`` is true (default; falls back to ``False``
    on CuPy-pinned-alloc failure), the input is opened as a raw ``os.open``
    fd and chunks are loaded via ``os.preadv`` into a ring of pinned host
    buffers, then async-copied to the GPU on non-blocking streams — the
    same pattern step1 (screen) and step4 (gwas) use via
    :class:`CugenReaderPinned`. Output bytes are staged through a matching
    pinned ring on a D2H stream before writing to disk. With ``n_buffers=2``
    chunk i+1's read can begin while chunk i is mid-decode on GPU.

    Returns wall time in seconds.
    """
    t_start = time.time()

    if not HAS_CUPY:
        raise RuntimeError("CuPy required for GPU subsetting")

    if use_pinned is None:
        use_pinned = os.environ.get('USE_PINNED_READER', '1') == '1'

    sample_indices = _resolve_sample_indices(sample_indices)
    n_subset = len(sample_indices)
    input_path = str(input_path)
    output_path = str(output_path)

    # Read source header
    with open(input_path, "rb") as f:
        header = f.read(HEADER_SIZE)

    magic = header[0:8]
    if magic != CUGEN_MAGIC:
        raise ValueError(f"Not a cugen file: {input_path}")

    src_n_samples = struct.unpack_from("<Q", header, 16)[0]
    n_variants = struct.unpack_from("<Q", header, 24)[0]
    src_bpv = struct.unpack_from("<Q", header, 32)[0]
    src_data_offset = struct.unpack_from("<Q", header, 48)[0]
    src_gidx_offset = struct.unpack_from("<Q", header, 56)[0]
    src_flags = struct.unpack_from("<I", header, 64)[0]

    if verbose:
        src_size = os.path.getsize(input_path)
        mode_tag = "pinned" if use_pinned else "pageable"
        print(
            f"  {os.path.basename(input_path)} [{mode_tag}]: "
            f"{src_n_samples:,} -> {n_subset:,} samples, "
            f"{n_variants:,} variants, {src_size/1e9:.1f}GB",
            flush=True,
        )

    if sample_indices.max() >= src_n_samples:
        raise IndexError(
            f"sample_indices.max()={sample_indices.max()} >= "
            f"src_n_samples={src_n_samples}"
        )
    if sample_indices.min() < 0:
        raise IndexError("sample_indices contains negative entries")

    # Output layout
    out_bpv = (n_subset + 3) // 4
    stats_size = n_variants * 4 * 3  # mu_x, sxx, maf
    gidx_size = n_variants * 8
    out_stats_offset = HEADER_SIZE
    out_gidx_offset = out_stats_offset + stats_size
    out_data_offset = out_gidx_offset + gidx_size

    # Read gidx (copy as-is)
    with open(input_path, "rb") as f:
        f.seek(src_gidx_offset)
        gidx = np.frombuffer(f.read(n_variants * 8), dtype=np.int64).copy()

    # Precompute byte/bit positions on GPU (tiny, stays resident)
    byte_indices_gpu = cp.asarray((sample_indices // 4).astype(np.int64))
    bit_shifts_gpu = cp.asarray((6 - 2 * (sample_indices % 4)).astype(np.uint8))

    # Stats buffers (CPU, written at end)
    all_mu_x = np.empty(n_variants, dtype=np.float32)
    all_sxx = np.empty(n_variants, dtype=np.float32)
    all_maf = np.empty(n_variants, dtype=np.float32)
    flags = FLAG_HAS_GIDX_MAP if (src_flags & FLAG_HAS_GIDX_MAP) else 0

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    # --- pinned / async I/O setup (mirrors CugenReaderPinned in io.py) ---
    chunk_in_bytes = chunk_size * src_bpv
    chunk_out_bytes = chunk_size * out_bpv

    pinned_in_mems = []
    pinned_in_views = []
    pinned_out_mems = []
    pinned_out_views = []
    h2d_streams = []
    d2h_streams = []

    if use_pinned:
        try:
            for _ in range(n_buffers):
                m_in = cp.cuda.alloc_pinned_memory(chunk_in_bytes)
                pinned_in_mems.append(m_in)
                pinned_in_views.append(
                    np.frombuffer(m_in, dtype=np.uint8, count=chunk_in_bytes)
                )
                m_out = cp.cuda.alloc_pinned_memory(chunk_out_bytes)
                pinned_out_mems.append(m_out)
                pinned_out_views.append(
                    np.frombuffer(m_out, dtype=np.uint8, count=chunk_out_bytes)
                )
                h2d_streams.append(cp.cuda.Stream(non_blocking=True))
                d2h_streams.append(cp.cuda.Stream(non_blocking=True))
        except Exception as e:  # noqa: BLE001 — fall back rather than crash
            if verbose:
                print(f"    [pinned alloc failed: {e}; falling back to pageable]",
                      flush=True)
            pinned_in_mems = pinned_in_views = pinned_out_mems = pinned_out_views = []
            h2d_streams = d2h_streams = []
            use_pinned = False

    t_io = 0.0
    t_gpu = 0.0

    if use_pinned:
        fin_fd = os.open(input_path, os.O_RDONLY)
    else:
        fin_fd = None

    try:
        with open(input_path, "rb") as fin, open(output_path, "wb") as fout:
            # Write placeholder for header + stats + gidx (filled at end)
            fout.write(b"\x00" * out_data_offset)

            for block_idx, block_start in enumerate(
                    range(0, n_variants, chunk_size)):
                block_end = min(block_start + chunk_size, n_variants)
                n_block = block_end - block_start
                n_in = n_block * src_bpv
                n_out = n_block * out_bpv
                buf_idx = block_idx % n_buffers if use_pinned else 0

                # --- I/O: Read packed bytes (pinned + async H2D or pageable) ---
                t0 = time.time()
                if use_pinned:
                    h2d_stream = h2d_streams[buf_idx]
                    # Wait for the prior use of this ring slot to finish before
                    # overwriting the pinned buffer (same shape as session-6
                    # ring discipline in CugenReaderPinned.read_to_gpu).
                    if block_idx >= n_buffers:
                        h2d_stream.synchronize()
                    view = memoryview(pinned_in_views[buf_idx])[:n_in]
                    n_read = os.preadv(
                        fin_fd, [view],
                        src_data_offset + block_start * src_bpv,
                    )
                    if n_read != n_in:
                        raise IOError(
                            f"preadv short read: got {n_read}, expected {n_in}"
                        )
                    packed_gpu_flat = cp.empty(n_in, dtype=cp.uint8)
                    cp.cuda.runtime.memcpyAsync(
                        packed_gpu_flat.data.ptr,
                        pinned_in_views[buf_idx].ctypes.data,
                        n_in,
                        cp.cuda.runtime.memcpyHostToDevice,
                        h2d_stream.ptr,
                    )
                    # Wait for THIS chunk's H2D before consuming it on GPU.
                    h2d_stream.synchronize()
                    packed_gpu = packed_gpu_flat.reshape(n_block, src_bpv)
                else:
                    fin.seek(src_data_offset + block_start * src_bpv)
                    packed_np = np.frombuffer(
                        fin.read(n_in), dtype=np.uint8
                    ).reshape(n_block, src_bpv)
                    packed_gpu = cp.asarray(packed_np)
                t_io += time.time() - t0

                # --- GPU: Extract + Stats + Repack ---
                t0 = time.time()
                # Extract target samples via fancy indexing
                needed = packed_gpu[:, byte_indices_gpu]  # (n_block, n_subset)
                genotypes = ((needed >> bit_shifts_gpu[None, :]) & 0x03).astype(cp.uint8)
                del needed, packed_gpu

                # Compute stats on GPU
                missing = (genotypes == 3)
                geno_f = genotypes.astype(cp.float32)
                geno_f[missing] = 0.0

                n_valid = cp.maximum(
                    (~missing).sum(axis=1).astype(cp.float32), 1.0
                )
                mu_x = geno_f.sum(axis=1) / n_valid

                # In-place centering (saves one full-size allocation)
                geno_f -= mu_x[:, None]
                geno_f[missing] = 0.0
                sxx = (geno_f ** 2).sum(axis=1)

                af = mu_x / 2.0
                maf = cp.minimum(af, 1.0 - af)

                if missing.any():
                    flags |= FLAG_HAS_MISSING

                # Copy stats to CPU
                all_mu_x[block_start:block_end] = cp.asnumpy(mu_x)
                all_sxx[block_start:block_end] = cp.asnumpy(sxx)
                all_maf[block_start:block_end] = cp.asnumpy(maf)

                del geno_f, missing, mu_x, sxx, maf, af, n_valid

                # Repack genotypes on GPU
                n_padded = out_bpv * 4
                if n_padded > n_subset:
                    padded = cp.zeros((n_block, n_padded), dtype=cp.uint8)
                    padded[:, :n_subset] = genotypes
                else:
                    padded = genotypes

                reshaped = padded.reshape(n_block, out_bpv, 4)
                packed_out = (
                    (reshaped[:, :, 0].astype(cp.uint8) << 6)
                    | (reshaped[:, :, 1].astype(cp.uint8) << 4)
                    | (reshaped[:, :, 2].astype(cp.uint8) << 2)
                    | reshaped[:, :, 3].astype(cp.uint8)
                )
                packed_out_flat = packed_out.ravel()
                t_gpu += time.time() - t0

                # --- I/O: Write packed output (pinned + async D2H or pageable) ---
                t0 = time.time()
                if use_pinned:
                    d2h_stream = d2h_streams[buf_idx]
                    # Same ring discipline on the write side.
                    if block_idx >= n_buffers:
                        d2h_stream.synchronize()
                    cp.cuda.runtime.memcpyAsync(
                        pinned_out_views[buf_idx].ctypes.data,
                        packed_out_flat.data.ptr,
                        n_out,
                        cp.cuda.runtime.memcpyDeviceToHost,
                        d2h_stream.ptr,
                    )
                    d2h_stream.synchronize()
                    fout.write(memoryview(pinned_out_views[buf_idx])[:n_out])
                else:
                    packed_out_np = cp.asnumpy(packed_out)
                    fout.write(packed_out_np.tobytes())
                t_io += time.time() - t0

                del genotypes, padded, reshaped, packed_out, packed_out_flat
                if not use_pinned:
                    del packed_out_np
                cp.get_default_memory_pool().free_all_blocks()

            # Write stats
            fout.seek(out_stats_offset)
            fout.write(all_mu_x.tobytes())
            fout.write(all_sxx.tobytes())
            fout.write(all_maf.tobytes())

            # Write gidx
            fout.seek(out_gidx_offset)
            fout.write(gidx.tobytes())

            # Write header
            fout.seek(0)
            out_header = bytearray(HEADER_SIZE)
            out_header[0:8] = CUGEN_MAGIC
            struct.pack_into("<I", out_header, 8, CUGEN_VERSION)
            struct.pack_into("<I", out_header, 12, ENCODING_2BIT)
            struct.pack_into("<Q", out_header, 16, n_subset)
            struct.pack_into("<Q", out_header, 24, n_variants)
            struct.pack_into("<Q", out_header, 32, out_bpv)
            struct.pack_into("<Q", out_header, 40, out_stats_offset)
            struct.pack_into("<Q", out_header, 48, out_data_offset)
            struct.pack_into("<Q", out_header, 56, out_gidx_offset)
            struct.pack_into("<I", out_header, 64, flags)
            fout.write(out_header)
    finally:
        if fin_fd is not None:
            try:
                os.close(fin_fd)
            except OSError:
                pass

    elapsed = time.time() - t_start
    out_size = os.path.getsize(output_path)

    if verbose:
        print(
            f"    -> {elapsed:.1f}s ({t_gpu:.1f}s GPU, {t_io:.1f}s I/O), "
            f"{out_size/1e9:.2f}GB, {n_variants/elapsed:.0f} var/s",
            flush=True,
        )

    return elapsed


def _subset_chrom_worker(payload):
    """Spawn-safe worker: each process gets one chr to subset.

    Receives ``(chrom, input_path, output_path, sample_indices_array,
    chunk_size, verbose, device, use_pinned)`` and calls
    :func:`subset_cugen_file`. Each worker initialises its own CuPy context
    (required for CUDA MPS).
    """
    (chrom, input_path, output_path, sample_indices, chunk_size,
     verbose, device, use_pinned) = payload
    import cupy as _cp
    _cp.cuda.Device(device).use()
    elapsed = subset_cugen_file(
        input_path, output_path, sample_indices,
        chunk_size=chunk_size, verbose=verbose, use_pinned=use_pinned,
    )
    return int(chrom), float(elapsed)


def subset_cugen_dir(
    source_dir: Union[str, Path],
    output_dir: Union[str, Path],
    sample_indices: SampleIndices,
    *,
    chromosomes: Sequence[int] = range(1, 23),
    chunk_size: int = 8192,
    verbose: bool = True,
    skip_existing: bool = False,
    n_workers: int = 1,
    device: int = 0,
    use_pinned: Optional[bool] = None,
) -> dict:
    """Batch-subset every ``chr{c}.cugen`` in ``source_dir`` into ``output_dir``.

    Parameters
    ----------
    source_dir : path
        Directory containing ``chr{1..22}.cugen``.
    output_dir : path
        Output directory. Created if missing.
    sample_indices : ndarray | path | (npz_path, key)
        Sample positions in the source cugen. Same conventions as
        :func:`subset_cugen_file`.
    chromosomes : iterable of int
        Chromosomes to process (default 1..22).
    chunk_size : int
        Variants per GPU chunk.
    skip_existing : bool
        If True, leave already-written output cugens in place. Useful for
        rerunning a partially-completed batch without redoing the work.
    n_workers : int
        Number of parallel processes (``multiprocessing.Pool(spawn)``).
        Default ``1`` = sequential (back-compat). Recommended ``8`` on H100
        SXM5 with CUDA MPS started in the SLURM wrapper. Each worker
        initialises its own CuPy context; per-chr work is embarrassingly
        parallel — the only shared resource is the GPU, so MPS makes the
        otherwise serialised CUDA contexts overlap.
    device : int
        GPU device index, passed to each worker (default ``0``).

    Returns
    -------
    dict
        ``{'n_subset': int, 'per_chr': {chrom: seconds}, 'total_s': float,
        'total_bytes': int, 'wall_s': float}``. ``total_s`` is the sum of
        per-chr times (sequential equivalent); ``wall_s`` is the actual
        wall-clock duration, which is what scales with ``n_workers``.
    """
    sample_indices = _resolve_sample_indices(sample_indices)
    n_subset = len(sample_indices)
    source_dir = str(source_dir)
    output_dir = str(output_dir)
    chromosomes = list(chromosomes)

    if verbose:
        print(f"Subset size: {n_subset:,} samples", flush=True)
        print(f"Source: {source_dir}", flush=True)
        print(f"Output: {output_dir}", flush=True)
        if n_workers > 1:
            print(f"Parallelism: {n_workers} worker processes (Pool(spawn))",
                  flush=True)

    os.makedirs(output_dir, exist_ok=True)
    per_chr: dict = {}
    total_time = 0.0
    wall_t0 = time.time()

    # Build job list, honouring skip_existing and missing-input.
    jobs = []
    for chrom in chromosomes:
        input_path = os.path.join(source_dir, f"chr{chrom}.cugen")
        out_path = os.path.join(output_dir, f"chr{chrom}.cugen")
        if not os.path.exists(input_path):
            if verbose:
                print(f"  WARNING: {input_path} not found, skipping", flush=True)
            continue
        if skip_existing and os.path.exists(out_path):
            if verbose:
                print(f"  [skip] chr{chrom} output exists", flush=True)
            per_chr[int(chrom)] = 0.0
            continue
        jobs.append((int(chrom), input_path, out_path))

    if n_workers <= 1 or len(jobs) <= 1:
        for chrom, input_path, out_path in jobs:
            elapsed = subset_cugen_file(
                input_path, out_path, sample_indices,
                chunk_size=chunk_size, verbose=verbose,
                use_pinned=use_pinned,
            )
            per_chr[chrom] = elapsed
            total_time += elapsed
    else:
        actual_workers = min(int(n_workers), len(jobs))
        payloads = [
            (chrom, input_path, out_path, sample_indices, chunk_size,
             verbose, device, use_pinned)
            for (chrom, input_path, out_path) in jobs
        ]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=actual_workers) as pool:
            for chrom, elapsed in pool.imap_unordered(
                    _subset_chrom_worker, payloads):
                per_chr[chrom] = elapsed
                total_time += elapsed

    wall_s = time.time() - wall_t0

    total_bytes = 0
    n_files = 0
    for f in os.listdir(output_dir):
        if f.endswith(".cugen"):
            n_files += 1
            total_bytes += os.path.getsize(os.path.join(output_dir, f))

    if verbose:
        print(
            f"\nTotal: {n_files}/{len(chromosomes)} chromosomes — "
            f"sum_per_chr={total_time:.1f}s, wall={wall_s:.1f}s "
            f"({total_bytes/1e9:.1f}GB)",
            flush=True,
        )

    return {
        "n_subset": n_subset,
        "per_chr": per_chr,
        "total_s": total_time,
        "wall_s": wall_s,
        "total_bytes": total_bytes,
    }


# ----------------------------------------------------------------------------
# PLINK / Hail aliases
# ----------------------------------------------------------------------------
def filter_cols(
    input_path,
    output_path,
    sample_indices,
    **kw,
):
    """Filter samples (cugen columns).

    PLINK ``--keep`` semantics. Dispatches on whether ``input_path`` is a
    file or directory: file → :func:`subset_cugen_file`, directory →
    :func:`subset_cugen_dir`. ``invert`` is not supported yet.
    """
    if kw.pop("invert", False):
        raise NotImplementedError(
            "filter_cols(invert=True) (PLINK --remove) requires the complement "
            "index — pass the precomputed indices yourself for now."
        )
    if os.path.isdir(str(input_path)):
        return subset_cugen_dir(input_path, output_path, sample_indices, **kw)
    return subset_cugen_file(input_path, output_path, sample_indices, **kw)


def filter_rows(*a, **kw):
    """Filter variants (cugen rows). PLINK ``--extract`` semantics.

    v0.2 roadmap — variant subset of a cugen is not yet implemented.
    """
    return _stub("subset.filter_rows")
