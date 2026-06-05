"""cugen.qc — variant and sample QC over .cugen files.

Two entry points:

``variant_qc(cugen_path_or_dir, ...) → DataFrame``
    Per-variant statistics + filter mask. Cheap when only header stats
    (mu_x / sxx / maf) are needed (no decode). Expensive when
    ``missing_rate_max`` or ``hwe_p_min`` is requested — those require a
    full chunked decode of the genotype matrix.

``sample_qc(cugen_dir, ...) → DataFrame``
    Per-sample missing rate + heterozygosity rate, computed by streaming
    every chr cugen through the GPU decoder.

Returned DataFrames have a boolean ``keep`` column that is True iff all
specified filter thresholds are passed. Optional ``output`` argument writes
TSV or feather; the ``keep`` mask doubles as input for ``cg.filter_cols``
(sample subset) and (eventually) ``cg.filter_rows`` (variant subset).

Computation primitives match ``subset.py``: 2-bit byte decode → 0/1/2/3
genotypes; missing = (geno==3); homhet = (geno==1); homalt = (geno==2).
Per-variant HWE is computed via Pearson chi² against expected counts under
HWE, with a small-cell guard (skip when expected count < 5 in any cell).

GPU streaming respects ``USE_PINNED_READER=1`` via the io.read_cugen swap.
"""
from __future__ import annotations

import math
import os
import struct
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:  # noqa: BLE001
    HAS_CUPY = False
    cp = None

from .io import (CUGEN_MAGIC, HEADER_SIZE, CugenReader, read_cugen,
                 read_cugen_header)


_CHUNK_VARIANTS = 4096


# ---------------------------------------------------------------------------
# Internal: decode a chunk and return geno-count tensors.
# ---------------------------------------------------------------------------
def _decode_chunk_counts(reader: CugenReader, start: int, end: int):
    """Returns (n_geno0, n_geno1, n_geno2, n_missing) per variant for the chunk.

    Re-decodes packed bytes on GPU; counts each 2-bit value independently
    instead of going through read_to_gpu (which maps geno=3 to 0).
    """
    bpv = reader.bytes_per_variant
    n_samples = reader.n_samples
    n_variants = end - start
    if n_variants == 0:
        z = cp.zeros(0, dtype=cp.int64)
        return z, z, z, z

    # Read raw packed bytes for the chunk
    raw = reader.read_packed_bytes(start, end)
    packed_np = np.frombuffer(raw, dtype=np.uint8).reshape(n_variants, bpv)
    packed_gpu = cp.asarray(packed_np)

    # Sample-position byte/bit lookup (cached on first call per reader)
    if not hasattr(reader, "_qc_byte_idx") or reader._qc_byte_idx is None:
        sample_idx = np.arange(n_samples, dtype=np.int64)
        reader._qc_byte_idx = cp.asarray((sample_idx // 4).astype(np.int64))
        reader._qc_bit_shift = cp.asarray(
            (6 - 2 * (sample_idx % 4)).astype(np.uint8)
        )
    byte_idx = reader._qc_byte_idx
    bit_shift = reader._qc_bit_shift

    # Decode: shape (n_variants, n_samples), values in {0,1,2,3}
    needed = packed_gpu[:, byte_idx]
    geno = ((needed >> bit_shift[None, :]) & 0x03).astype(cp.uint8)
    del packed_gpu, needed

    # Per-variant counts of each genotype code
    n_geno0 = (geno == 0).sum(axis=1).astype(cp.int64)
    n_geno1 = (geno == 1).sum(axis=1).astype(cp.int64)
    n_geno2 = (geno == 2).sum(axis=1).astype(cp.int64)
    n_missing = (geno == 3).sum(axis=1).astype(cp.int64)
    return n_geno0, n_geno1, n_geno2, n_missing


def _hwe_chi2(n0: 'cp.ndarray', n1: 'cp.ndarray',
              n2: 'cp.ndarray') -> 'cp.ndarray':
    """Pearson chi² for HWE. Returns per-variant chi² (cp.float32);
    invalid (expected < 5 in any cell) → NaN.
    """
    n = (n0 + n1 + n2).astype(cp.float32)
    p = (2.0 * n2 + n1) / cp.maximum(2.0 * n, 1.0)
    q = 1.0 - p
    exp0 = n * q * q
    exp1 = 2.0 * n * p * q
    exp2 = n * p * p
    valid = (exp0 >= 5) & (exp1 >= 5) & (exp2 >= 5)
    o0 = n0.astype(cp.float32)
    o1 = n1.astype(cp.float32)
    o2 = n2.astype(cp.float32)
    chi2 = cp.where(
        valid,
        (o0 - exp0) ** 2 / cp.maximum(exp0, 1e-10)
        + (o1 - exp1) ** 2 / cp.maximum(exp1, 1e-10)
        + (o2 - exp2) ** 2 / cp.maximum(exp2, 1e-10),
        cp.nan,
    )
    return chi2


def _chi2_p_1df(chi2: 'cp.ndarray') -> 'cp.ndarray':
    """Right-tail p-value for chi² with 1 df: ``1 - erf(sqrt(chi2/2))``."""
    # cp.special.erf(x) returns elementwise erf
    from cupyx.scipy import special as _sp  # noqa: PLC0415
    return 1.0 - _sp.erf(cp.sqrt(cp.maximum(chi2, 0.0) / 2.0))


# ---------------------------------------------------------------------------
# Public: variant_qc
# ---------------------------------------------------------------------------
def variant_qc(
    cugen: Union[str, Path],
    *,
    chromosomes: Optional[Sequence[int]] = None,
    cugen_pattern: str = "chr{chrom}.cugen",
    maf_min: Optional[float] = None,
    maf_max: Optional[float] = None,
    missing_rate_max: Optional[float] = None,
    hwe_p_min: Optional[float] = None,
    output: Optional[Union[str, Path]] = None,
    chunk_size: int = _CHUNK_VARIANTS,
    device: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute per-variant QC metrics and a filter mask.

    Parameters
    ----------
    cugen
        Either a single ``.cugen`` file path OR a directory containing
        ``chr{1..22}.cugen`` files.
    chromosomes
        When ``cugen`` is a directory, restrict to these chromosomes (default
        1..22). Ignored for a single-file input.
    maf_min, maf_max, missing_rate_max, hwe_p_min
        Filters applied to the ``keep`` column.
        * ``maf_min`` / ``maf_max`` use the cugen header's pre-computed MAF
          (no decode required).
        * ``missing_rate_max`` and ``hwe_p_min`` REQUIRE full decode of every
          variant → expect ~10-30 s per chr on H100 for unified 336K.
    output
        Optional TSV / feather path. ``.tsv.gz`` triggers gzip.
    chunk_size
        Variants per GPU decode block (only matters when decoding).

    Returns
    -------
    DataFrame indexed by [chrom, gidx] with columns:
        mu_x, sxx, maf, n_geno0, n_geno1, n_geno2, n_missing,
        missing_rate, hwe_chi2, hwe_p, keep
        (decode-only columns set to NaN when fast-path is taken).
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for cg.variant_qc")
    cp.cuda.Device(device).use()

    need_decode = (missing_rate_max is not None) or (hwe_p_min is not None)
    cugen_str = str(cugen)
    if os.path.isdir(cugen_str):
        chrs = list(chromosomes) if chromosomes is not None else list(range(1, 23))
        files = []
        for c in chrs:
            p = os.path.join(cugen_str, cugen_pattern.format(chrom=c))
            if os.path.exists(p):
                files.append((c, p))
            elif verbose:
                print(f"[variant_qc] WARNING: {p} missing", flush=True)
    else:
        # single file — use 0 as chrom placeholder unless header carries it
        files = [(0, cugen_str)]

    rows = []
    for chrom, path in files:
        reader = read_cugen(path, device=device)
        try:
            n_v = reader.n_variants
            mu_x = reader.mu_x.astype(np.float32, copy=False)
            sxx = reader.sxx.astype(np.float32, copy=False)
            maf = reader.maf.astype(np.float32, copy=False)

            base = {
                "chrom": np.full(n_v, chrom, dtype=np.int32),
                "gidx": reader.gidx.astype(np.int64, copy=False),
                "mu_x": mu_x,
                "sxx": sxx,
                "maf": maf,
            }
            if need_decode:
                n_geno0 = np.empty(n_v, dtype=np.int64)
                n_geno1 = np.empty(n_v, dtype=np.int64)
                n_geno2 = np.empty(n_v, dtype=np.int64)
                n_missing = np.empty(n_v, dtype=np.int64)
                hwe_chi2 = np.empty(n_v, dtype=np.float32)
                for s in range(0, n_v, chunk_size):
                    e = min(s + chunk_size, n_v)
                    g0, g1, g2, gm = _decode_chunk_counts(reader, s, e)
                    n_geno0[s:e] = cp.asnumpy(g0)
                    n_geno1[s:e] = cp.asnumpy(g1)
                    n_geno2[s:e] = cp.asnumpy(g2)
                    n_missing[s:e] = cp.asnumpy(gm)
                    chi2 = _hwe_chi2(g0, g1, g2)
                    hwe_chi2[s:e] = cp.asnumpy(chi2)
                base.update({
                    "n_geno0": n_geno0,
                    "n_geno1": n_geno1,
                    "n_geno2": n_geno2,
                    "n_missing": n_missing,
                    "missing_rate": n_missing / float(reader.n_samples),
                    "hwe_chi2": hwe_chi2,
                })
                # p-value (CPU side, scalar)
                from scipy.stats import chi2 as _chi2  # noqa: PLC0415
                base["hwe_p"] = np.where(
                    np.isnan(hwe_chi2),
                    np.nan,
                    1.0 - _chi2.cdf(hwe_chi2, df=1),
                )
            else:
                # placeholders
                nan_f = np.full(n_v, np.nan, dtype=np.float32)
                base.update({
                    "n_geno0": np.full(n_v, -1, dtype=np.int64),
                    "n_geno1": np.full(n_v, -1, dtype=np.int64),
                    "n_geno2": np.full(n_v, -1, dtype=np.int64),
                    "n_missing": np.full(n_v, -1, dtype=np.int64),
                    "missing_rate": nan_f,
                    "hwe_chi2": nan_f,
                    "hwe_p": nan_f,
                })
            rows.append(pd.DataFrame(base))
            if verbose:
                tag = "decoded" if need_decode else "fast"
                print(f"[variant_qc] chr{chrom}: {n_v:,} variants ({tag})",
                      flush=True)
        finally:
            reader.close()

    df = pd.concat(rows, ignore_index=True)
    keep = np.ones(len(df), dtype=bool)
    if maf_min is not None:
        keep &= (df["maf"].values >= maf_min)
    if maf_max is not None:
        keep &= (df["maf"].values <= maf_max)
    if missing_rate_max is not None:
        keep &= (df["missing_rate"].values <= missing_rate_max)
    if hwe_p_min is not None:
        keep &= (df["hwe_p"].values >= hwe_p_min) | np.isnan(df["hwe_p"].values)
    df["keep"] = keep

    if verbose:
        print(f"[variant_qc] kept {int(keep.sum()):,} / {len(df):,} variants",
              flush=True)

    if output is not None:
        _write_df(df, str(output))
    return df


# ---------------------------------------------------------------------------
# Public: sample_qc
# ---------------------------------------------------------------------------
def sample_qc(
    cugen_dir: Union[str, Path],
    *,
    chromosomes: Sequence[int] = range(1, 23),
    cugen_pattern: str = "chr{chrom}.cugen",
    missing_rate_max: Optional[float] = None,
    het_rate_min: Optional[float] = None,
    het_rate_max: Optional[float] = None,
    fid_order: Optional[Sequence] = None,
    output: Optional[Union[str, Path]] = None,
    chunk_size: int = _CHUNK_VARIANTS,
    device: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Per-sample missing rate + heterozygosity.

    Streams every chr cugen, accumulates per-sample missing-genotype and
    het-genotype counts, then divides by the total number of variants seen.

    Parameters
    ----------
    cugen_dir
        Directory containing chr{1..22}.cugen.
    missing_rate_max, het_rate_min, het_rate_max
        Optional filter thresholds populating the ``keep`` column.
    fid_order
        Optional iterable of sample IDs to label rows.
    chunk_size
        Variants per GPU decode block.

    Returns
    -------
    DataFrame with columns: sample_idx (or sample_id), n_total, n_missing,
        n_het, missing_rate, het_rate, keep.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for cg.sample_qc")
    cp.cuda.Device(device).use()
    cugen_dir = str(cugen_dir)
    chrs = list(chromosomes)

    n_samples = None
    n_missing_gpu = None
    n_het_gpu = None
    total_variants = 0

    for chrom in chrs:
        path = os.path.join(cugen_dir, cugen_pattern.format(chrom=chrom))
        if not os.path.exists(path):
            if verbose:
                print(f"[sample_qc] WARNING: {path} missing, skipping",
                      flush=True)
            continue
        reader = read_cugen(path, device=device)
        try:
            if n_samples is None:
                n_samples = reader.n_samples
                n_missing_gpu = cp.zeros(n_samples, dtype=cp.int64)
                n_het_gpu = cp.zeros(n_samples, dtype=cp.int64)
            elif reader.n_samples != n_samples:
                raise ValueError(
                    f"sample count mismatch on chr{chrom}: "
                    f"{reader.n_samples} vs {n_samples}"
                )
            n_v = reader.n_variants
            bpv = reader.bytes_per_variant
            # Sample byte/bit lookup, cached on reader
            if not hasattr(reader, "_qc_byte_idx") or reader._qc_byte_idx is None:
                sample_idx = np.arange(n_samples, dtype=np.int64)
                reader._qc_byte_idx = cp.asarray((sample_idx // 4).astype(np.int64))
                reader._qc_bit_shift = cp.asarray(
                    (6 - 2 * (sample_idx % 4)).astype(np.uint8)
                )
            byte_idx = reader._qc_byte_idx
            bit_shift = reader._qc_bit_shift

            for s in range(0, n_v, chunk_size):
                e = min(s + chunk_size, n_v)
                raw = reader.read_packed_bytes(s, e)
                packed_np = np.frombuffer(raw, dtype=np.uint8).reshape(e - s, bpv)
                packed_gpu = cp.asarray(packed_np)
                needed = packed_gpu[:, byte_idx]
                geno = ((needed >> bit_shift[None, :]) & 0x03).astype(cp.uint8)
                del packed_gpu, needed
                # Per-sample sums over the variants in this chunk
                n_missing_gpu += (geno == 3).sum(axis=0).astype(cp.int64)
                n_het_gpu += (geno == 1).sum(axis=0).astype(cp.int64)
                del geno
            total_variants += n_v
            if verbose:
                print(f"[sample_qc] chr{chrom}: +{n_v:,} variants "
                      f"(total {total_variants:,})", flush=True)
            cp.get_default_memory_pool().free_all_blocks()
        finally:
            reader.close()

    if n_samples is None:
        raise RuntimeError("no chromosomes processed — check cugen_dir")

    n_missing = cp.asnumpy(n_missing_gpu)
    n_het = cp.asnumpy(n_het_gpu)
    missing_rate = n_missing / float(total_variants)
    het_rate = n_het / float(total_variants)

    if fid_order is not None:
        if len(fid_order) != n_samples:
            raise ValueError(
                f"fid_order length {len(fid_order)} != n_samples {n_samples}"
            )
        df = pd.DataFrame({
            "sample_id": list(fid_order),
            "n_total": total_variants,
            "n_missing": n_missing,
            "n_het": n_het,
            "missing_rate": missing_rate,
            "het_rate": het_rate,
        })
    else:
        df = pd.DataFrame({
            "sample_idx": np.arange(n_samples),
            "n_total": total_variants,
            "n_missing": n_missing,
            "n_het": n_het,
            "missing_rate": missing_rate,
            "het_rate": het_rate,
        })

    keep = np.ones(n_samples, dtype=bool)
    if missing_rate_max is not None:
        keep &= (missing_rate <= missing_rate_max)
    if het_rate_min is not None:
        keep &= (het_rate >= het_rate_min)
    if het_rate_max is not None:
        keep &= (het_rate <= het_rate_max)
    df["keep"] = keep

    if verbose:
        print(f"[sample_qc] kept {int(keep.sum()):,} / {n_samples:,} samples",
              flush=True)

    if output is not None:
        _write_df(df, str(output))
    return df


def _write_df(df: pd.DataFrame, path: str) -> None:
    if path.endswith(".feather"):
        df.to_feather(path)
    else:
        sep = "\t" if path.endswith((".tsv", ".tsv.gz")) else ","
        compression = "gzip" if path.endswith(".gz") else None
        df.to_csv(path, sep=sep, index=False, compression=compression)
