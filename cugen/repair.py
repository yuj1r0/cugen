"""cugen.repair — recompute & rewrite a .cugen file's stored stats block.

The value of the .cugen format is *trustworthy precomputed stats*: every
downstream primitive (screen, LASSO, association) divides by the stored
per-variant ``sxx`` and reads the stored ``maf``. If a subset of variants has a
corrupt stats block — e.g. ``sxx≈0`` / ``maf==0`` for a *polymorphic* variant
(a stats-build artifact observed session 50 in the array cugens, a contiguous
gidx tail) — then ``beta = num / sxx`` blows up to ~1e12 and poisons an
all-variant GWAS (the falsely-"polygenic" null QQ band). PRS/LASSO is unaffected
(it never selects a maf=0 variant) and production GWAS output is unaffected
(maf_min filters them out) — only the diagnostic all-variant null exposed it.

The fix is NOT to recompute stats inside the association (that would defeat the
whole point of precomputed stats). It is to **repair the stored block once** so
the trusted stats are correct, then trust them everywhere. This module does
exactly that: it re-derives ``mu_x / sxx / maf`` from the file's own decoded
2-bit genotypes — using the identical missing-aware math as
:func:`cugen.subset.subset_cugen_file` (the canonical cugen stats recompute) —
and overwrites the 3-float stats block IN PLACE at ``stats_offset``. The
genotype/gidx blocks are untouched; only ~``3·n_variants·4`` bytes are rewritten
per file. Cost is the one full genotype read (a few minutes for an array cugen).

Public API
----------
``repair_cugen_file(path, **kw)`` -> dict
    Recompute & rewrite one file's stats. Returns a change report.
``repair_cugen_dir(source_dir, **kw)`` -> dict
    Repair every ``chr{1..22}.cugen`` in a directory (spawn Pool, one chr/worker,
    CUDA MPS — same parallelism as :func:`cugen.subset.subset_cugen_dir`).

Both accept ``dry_run=True`` (report what WOULD change, write nothing) and
``backup=True`` (default; save the old stats block to ``<path>.stats_bak``
before the first in-place write, so a repair is reversible).
"""

import multiprocessing as mp
import os
import struct
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None

from .io import CUGEN_MAGIC, HEADER_SIZE


def _recompute_stats_gpu(packed_gpu, byte_indices_gpu, bit_shifts_gpu):
    """Decode a (n_block, bytes_per_variant) packed block for ALL samples and
    return (mu_x, sxx, maf, any_missing) — byte-identical math to
    subset.py:274-305 (missing-aware mean, centered SS, maf=min(af,1-af))."""
    # Gather every sample's byte then extract its 2-bit code (variant-major).
    needed = packed_gpu[:, byte_indices_gpu]                 # (n_block, n_samples)
    genotypes = ((needed >> bit_shifts_gpu[None, :]) & 0x03).astype(cp.uint8)
    del needed

    missing = (genotypes == 3)
    geno_f = genotypes.astype(cp.float32)
    geno_f[missing] = 0.0

    n_valid = cp.maximum((~missing).sum(axis=1).astype(cp.float32), 1.0)
    mu_x = geno_f.sum(axis=1) / n_valid

    geno_f -= mu_x[:, None]                                  # in-place centering
    geno_f[missing] = 0.0
    sxx = (geno_f ** 2).sum(axis=1)

    af = mu_x / 2.0
    maf = cp.minimum(af, 1.0 - af)
    any_missing = bool(missing.any())

    del genotypes, geno_f, missing, n_valid, af
    return mu_x, sxx, maf, any_missing


def repair_cugen_file(
    path: Union[str, Path],
    *,
    chunk_size: int = 1024,
    backup: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    device: int = 0,
) -> dict:
    """Recompute ``mu_x / sxx / maf`` from decoded genotypes and rewrite the
    stats block of ``path`` in place.

    A variant is flagged 'repaired' when the recomputed stats differ materially
    from the stored ones — specifically ``|sxx_new - sxx_old| > tol·max(sxx_new,1)``
    or a stored ``maf==0`` with a polymorphic recompute. The classic corruption
    (stored ``sxx≈0`` for a common variant) shows up as a large relative Δsxx.

    Returns ``{path, n_variants, n_repaired, n_sxx_zeroed, max_abs_dsxx,
    max_rel_dsxx, elapsed_s, dry_run}``.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for cugen repair")
    cp.cuda.Device(device).use()

    t0 = time.time()
    path = str(path)

    with open(path, "rb") as f:
        header = f.read(HEADER_SIZE)
    if header[0:8] != CUGEN_MAGIC:
        raise ValueError(f"Not a cugen file: {path}")
    n_samples = struct.unpack_from("<Q", header, 16)[0]
    n_variants = struct.unpack_from("<Q", header, 24)[0]
    bpv = struct.unpack_from("<Q", header, 32)[0]
    stats_offset = struct.unpack_from("<Q", header, 40)[0]
    data_offset = struct.unpack_from("<Q", header, 48)[0]

    # Full-sample decode positions (identity — every sample kept).
    sample_idx = np.arange(n_samples, dtype=np.int64)
    byte_indices_gpu = cp.asarray((sample_idx // 4).astype(np.int64))
    bit_shifts_gpu = cp.asarray((6 - 2 * (sample_idx % 4)).astype(np.uint8))

    new_mu = np.empty(n_variants, dtype=np.float32)
    new_sxx = np.empty(n_variants, dtype=np.float32)
    new_maf = np.empty(n_variants, dtype=np.float32)

    fd = os.open(path, os.O_RDWR)
    try:
        # Read the current stats block first (for the change report + backup).
        old_blob = os.pread(fd, n_variants * 4 * 3, stats_offset)
        old_sxx = np.frombuffer(old_blob, dtype=np.float32,
                                count=n_variants, offset=n_variants * 4).copy()
        old_maf = np.frombuffer(old_blob, dtype=np.float32,
                                count=n_variants, offset=n_variants * 8).copy()

        for start in range(0, n_variants, chunk_size):
            end = min(start + chunk_size, n_variants)
            n_block = end - start
            n_in = n_block * bpv
            raw = os.pread(fd, n_in, data_offset + start * bpv)
            if len(raw) != n_in:
                raise IOError(f"short read at variant {start}: "
                              f"{len(raw)} != {n_in}")
            packed_gpu = cp.asarray(
                np.frombuffer(raw, dtype=np.uint8).reshape(n_block, bpv))
            mu_x, sxx, maf, _ = _recompute_stats_gpu(
                packed_gpu, byte_indices_gpu, bit_shifts_gpu)
            new_mu[start:end] = cp.asnumpy(mu_x)
            new_sxx[start:end] = cp.asnumpy(sxx)
            new_maf[start:end] = cp.asnumpy(maf)
            del packed_gpu, mu_x, sxx, maf
            cp.get_default_memory_pool().free_all_blocks()

        # Change report.
        d_sxx = np.abs(new_sxx - old_sxx)
        denom = np.maximum(new_sxx, 1.0)
        rel = d_sxx / denom
        repaired = rel > 1e-3
        n_repaired = int(repaired.sum())
        n_sxx_zeroed = int(((old_sxx < 1e-6) & (new_sxx > 1.0)).sum())
        max_abs = float(d_sxx.max()) if n_variants else 0.0
        max_rel = float(rel.max()) if n_variants else 0.0

        if verbose:
            base = os.path.basename(path)
            print(f"  [repair] {base}: n_var={n_variants:,} "
                  f"repaired={n_repaired:,} sxx≈0→polymorphic={n_sxx_zeroed:,} "
                  f"max|Δsxx|={max_abs:.3g} max_rel={max_rel:.3g}"
                  f"{'  (DRY RUN, no write)' if dry_run else ''}", flush=True)

        if not dry_run:
            if backup:
                bak = path + ".stats_bak"
                if not os.path.exists(bak):
                    with open(bak, "wb") as fb:
                        fb.write(old_blob)
            new_blob = (new_mu.tobytes() + new_sxx.tobytes()
                        + new_maf.tobytes())
            n_written = os.pwrite(fd, new_blob, stats_offset)
            if n_written != len(new_blob):
                raise IOError(f"short stats write: {n_written} != {len(new_blob)}")
    finally:
        os.close(fd)

    return {
        "path": path,
        "n_variants": int(n_variants),
        "n_repaired": n_repaired,
        "n_sxx_zeroed": n_sxx_zeroed,
        "max_abs_dsxx": max_abs,
        "max_rel_dsxx": max_rel,
        "elapsed_s": time.time() - t0,
        "dry_run": bool(dry_run),
    }


def _repair_chrom_worker(payload):
    (chrom, path, chunk_size, backup, dry_run, verbose, device) = payload
    import cupy as _cp
    _cp.cuda.Device(device).use()
    rep = repair_cugen_file(
        path, chunk_size=chunk_size, backup=backup, dry_run=dry_run,
        verbose=verbose, device=device,
    )
    return int(chrom), rep


def repair_cugen_dir(
    source_dir: Union[str, Path],
    *,
    chromosomes: Sequence[int] = range(1, 23),
    chunk_size: int = 1024,
    backup: bool = True,
    dry_run: bool = False,
    verbose: bool = True,
    n_workers: int = 8,
    device: int = 0,
) -> dict:
    """Repair the stats block of every ``chr{c}.cugen`` in ``source_dir`` in
    place. Spawn Pool, one chromosome per worker (CUDA MPS) — the same
    parallelism as :func:`cugen.subset.subset_cugen_dir`, so 8 workers overlap
    the per-chr genotype reads. Returns ``{'per_chr': {chrom: report}, ...}``.
    """
    source_dir = str(source_dir)
    chromosomes = list(chromosomes)

    jobs = []
    for chrom in chromosomes:
        p = os.path.join(source_dir, f"chr{chrom}.cugen")
        if not os.path.exists(p):
            if verbose:
                print(f"  WARNING: {p} not found, skipping", flush=True)
            continue
        jobs.append((int(chrom), p))

    if verbose:
        print(f"[repair] {source_dir}: {len(jobs)} chr, "
              f"{'DRY RUN' if dry_run else 'in-place'}, "
              f"{n_workers} worker(s)", flush=True)

    wall_t0 = time.time()
    per_chr: dict = {}

    if n_workers <= 1 or len(jobs) <= 1:
        for chrom, p in jobs:
            _, rep = _repair_chrom_worker(
                (chrom, p, chunk_size, backup, dry_run, verbose, device))
            per_chr[chrom] = rep
    else:
        actual_workers = min(int(n_workers), len(jobs))
        payloads = [(chrom, p, chunk_size, backup, dry_run, verbose, device)
                    for (chrom, p) in jobs]
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=actual_workers) as pool:
            for chrom, rep in pool.imap_unordered(_repair_chrom_worker, payloads):
                per_chr[chrom] = rep

    total_repaired = sum(r["n_repaired"] for r in per_chr.values())
    total_zeroed = sum(r["n_sxx_zeroed"] for r in per_chr.values())
    total_var = sum(r["n_variants"] for r in per_chr.values())
    wall_s = time.time() - wall_t0

    if verbose:
        print(f"[repair] DONE: {len(per_chr)} chr, {total_var:,} variants, "
              f"{total_repaired:,} repaired ({total_zeroed:,} sxx≈0→polymorphic), "
              f"wall={wall_s:.1f}s", flush=True)

    return {
        "source_dir": source_dir,
        "per_chr": per_chr,
        "n_variants": total_var,
        "n_repaired": total_repaired,
        "n_sxx_zeroed": total_zeroed,
        "wall_s": wall_s,
        "dry_run": bool(dry_run),
    }


# PLINK/Hail-ish alias: repair dispatches on file-vs-dir.
def repair(target: Union[str, Path], **kw) -> dict:
    """Repair a single ``.cugen`` file or every ``chr*.cugen`` in a directory
    (dispatches on ``os.path.isdir``). See :func:`repair_cugen_file` /
    :func:`repair_cugen_dir` for keyword arguments."""
    target = str(target)
    if os.path.isdir(target):
        return repair_cugen_dir(target, **kw)
    return repair_cugen_file(target, **kw)
