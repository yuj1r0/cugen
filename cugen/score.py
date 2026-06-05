"""cugen.score — GPU polygenic scoring.

One-call PRS over a cugen directory::

    import cugen as cg

    prs = cg.score(
        weights="weights.feather",            # gidx + beta
        cugen_dir="/path/to/cugen_dir",
        output="prs.tsv",                     # optional
    )
    # prs is a DataFrame: sample_idx (0..n_samples-1), prs

Weights forms accepted:
  * DataFrame with ``gidx`` and ``beta`` (or ``BETA``) columns
  * dict ``{gidx: beta}``
  * path to feather, TSV, CSV, or .npz (with arrays ``gidx`` + ``beta``)

The intersection (cugen.gidx ∩ weights.gidx) is taken per chromosome; missing
variants contribute zero. Per-chr decode strategy auto-selects:
  * **chunked decode + mask** when weights cover ≥ 10 % of the chr's variants
    (more efficient at high density — same I/O regardless of match rate);
  * **batched random access** when weights are sparse (< 10 % density;
    coalesces adjacent indices into single mmap reads).

Mean-imputation of missing genotypes follows ``CugenReader.read_to_gpu``
semantics (geno=3 → 0 after centering, i.e. mean-impute on the
already-centered scale). The cugen's per-variant mu_x is subtracted before
the matmul so the PRS is a sum of centered-and-weighted dosages, matching
plink2 ``--score sum`` with center+mean-impute.

Returns a pandas DataFrame; also writes to ``output`` (TSV or feather by
extension) if provided. Wall time on an H100 SXM5 for 22-chr unified 336K
× ~750 K PRS-CS weights is typically <1 min.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:  # noqa: BLE001
    HAS_CUPY = False
    cp = None

from .io import CugenReader, read_cugen


_DENSITY_THRESHOLD = 0.10  # ≥10 % matched → chunked-load + mask
_CHUNK_SIZE = 4096         # variants per chunked-decode block


def _resolve_weights(weights) -> Dict[int, float]:
    """Accept DataFrame / dict / path; return ``{gidx: beta}`` dict."""
    if isinstance(weights, dict):
        return {int(k): float(v) for k, v in weights.items()}
    if isinstance(weights, pd.DataFrame):
        df = weights
    elif isinstance(weights, (str, Path)):
        p = str(weights)
        if p.endswith(".feather"):
            df = pd.read_feather(p)
        elif p.endswith(".npz"):
            with np.load(p) as z:
                return dict(zip(z["gidx"].astype(np.int64).tolist(),
                                z["beta"].astype(np.float64).tolist()))
        else:
            sep = "\t" if p.endswith((".tsv", ".tsv.gz")) else ","
            df = pd.read_csv(p, sep=sep)
    else:
        raise TypeError(
            f"weights must be DataFrame, dict, or path; got {type(weights)}"
        )

    if "gidx" not in df.columns:
        raise ValueError(f"weights missing 'gidx' column: {df.columns.tolist()}")
    beta_col = None
    for c in ("beta", "BETA", "weight", "WEIGHT"):
        if c in df.columns:
            beta_col = c
            break
    if beta_col is None:
        raise ValueError(
            f"weights missing 'beta'/'BETA'/'weight' column: "
            f"{df.columns.tolist()}"
        )
    return dict(zip(df["gidx"].astype(np.int64).tolist(),
                    df[beta_col].astype(np.float64).tolist()))


def _score_chr_dense(reader: CugenReader, local_idx: np.ndarray,
                     beta_chunk: np.ndarray, prs: 'cp.ndarray',
                     mu_subtract: bool, chunk_size: int) -> None:
    """Chunked decode + mask: read every variant in `chunk_size` blocks,
    mask to those in `local_idx`, mat-mul beta into `prs` in place.
    """
    n_v = reader.n_variants
    sort_order = np.argsort(local_idx)
    sorted_idx = local_idx[sort_order]
    sorted_beta = beta_chunk[sort_order]
    if mu_subtract:
        mu_all = reader.mu_x
    pos = 0  # cursor into sorted_idx
    for start in range(0, n_v, chunk_size):
        end = min(start + chunk_size, n_v)
        # Indices in this chunk that we care about
        lo = pos
        while pos < len(sorted_idx) and sorted_idx[pos] < end:
            pos += 1
        if pos == lo:
            continue  # no matches in this chunk
        chunk_idx_local = sorted_idx[lo:pos] - start  # offset within chunk
        chunk_beta = sorted_beta[lo:pos]

        X = reader.read_to_gpu(start, end)  # (n_samples, chunk_size)
        X_sub = X[:, cp.asarray(chunk_idx_local)]
        if mu_subtract:
            mu = cp.asarray(mu_all[start + chunk_idx_local], dtype=cp.float32)
            X_sub = X_sub - mu[None, :]
        beta_gpu = cp.asarray(chunk_beta, dtype=cp.float32)
        prs += X_sub @ beta_gpu
        del X, X_sub, beta_gpu
        cp.get_default_memory_pool().free_all_blocks()


def _score_chr_sparse(reader: CugenReader, local_idx: np.ndarray,
                      beta_chunk: np.ndarray, prs: 'cp.ndarray',
                      mu_subtract: bool) -> None:
    """Batched random-access read for sparse-weight cases."""
    X = reader.read_indices_to_gpu_batched(local_idx)  # (n_samples, k)
    if mu_subtract:
        mu = cp.asarray(reader.mu_x[local_idx], dtype=cp.float32)
        X = X - mu[None, :]
    beta_gpu = cp.asarray(beta_chunk, dtype=cp.float32)
    prs += X @ beta_gpu
    del X, beta_gpu
    cp.get_default_memory_pool().free_all_blocks()


def score(
    weights,
    cugen_dir: Union[str, Path],
    *,
    chromosomes: Sequence[int] = range(1, 23),
    cugen_pattern: str = "chr{chrom}.cugen",
    output: Optional[Union[str, Path]] = None,
    fid_order: Optional[Sequence] = None,
    center: bool = True,
    device: int = 0,
    chunk_size: int = _CHUNK_SIZE,
    density_threshold: float = _DENSITY_THRESHOLD,
    verbose: bool = True,
) -> pd.DataFrame:
    """Compute polygenic scores from a weights file + cugen directory.

    Parameters
    ----------
    weights
        DataFrame (gidx + beta), dict {gidx: beta}, or path (feather / TSV /
        CSV / NPZ).
    cugen_dir
        Directory containing per-chr cugens.
    chromosomes
        Iterable of chromosome numbers to score over (default 1..22).
    cugen_pattern
        Filename template; defaults to ``"chr{chrom}.cugen"``.
    output
        Optional path; emits TSV (``.tsv`` / ``.tsv.gz``) or feather (``.feather``).
    fid_order
        Optional iterable of sample IDs to label the output rows; otherwise
        rows are numbered 0..n_samples-1.
    center
        If True (default), subtract per-variant mu_x before the matmul so
        the score is a sum of centered dosages (plink2 ``--score sum --center``
        equivalent). If False, returns the raw sum (sensitive to allele coding).
    device
        GPU device index.
    chunk_size, density_threshold
        Tuning knobs for the per-chr decode strategy. ``density_threshold``
        triggers chunked decode + mask when (#matched / #chr_variants) ≥
        this value; otherwise uses batched random access.
    """
    if not HAS_CUPY:
        raise RuntimeError("CuPy required for cg.score")

    cp.cuda.Device(device).use()
    w = _resolve_weights(weights)
    if not w:
        raise ValueError("weights is empty")
    if verbose:
        print(f"[score] {len(w):,} weights loaded", flush=True)

    cugen_dir = str(cugen_dir)
    n_samples = None
    prs_gpu = None
    total_used = 0
    per_chr_stats = []

    for chrom in chromosomes:
        path = os.path.join(cugen_dir, cugen_pattern.format(chrom=chrom))
        if not os.path.exists(path):
            if verbose:
                print(f"[score] WARNING: {path} not found, skipping",
                      flush=True)
            continue
        reader = read_cugen(path, device=device)
        try:
            if n_samples is None:
                n_samples = reader.n_samples
                prs_gpu = cp.zeros(n_samples, dtype=cp.float32)
            elif reader.n_samples != n_samples:
                raise ValueError(
                    f"sample count mismatch on chr{chrom}: "
                    f"{reader.n_samples} vs {n_samples}"
                )

            # Match this chr's gidx into weights
            chr_gidx = reader.gidx
            mask = np.fromiter((g in w for g in chr_gidx.tolist()),
                               dtype=bool, count=len(chr_gidx))
            local_idx = np.flatnonzero(mask).astype(np.int64)
            if len(local_idx) == 0:
                per_chr_stats.append((chrom, 0, "skip"))
                continue
            beta_chunk = np.array(
                [w[int(g)] for g in chr_gidx[local_idx].tolist()],
                dtype=np.float64,
            )
            density = len(local_idx) / max(1, reader.n_variants)
            strategy = "dense" if density >= density_threshold else "sparse"
            if strategy == "dense":
                _score_chr_dense(reader, local_idx, beta_chunk, prs_gpu,
                                 center, chunk_size)
            else:
                _score_chr_sparse(reader, local_idx, beta_chunk, prs_gpu,
                                  center)
            total_used += len(local_idx)
            per_chr_stats.append((chrom, len(local_idx), strategy))
            if verbose:
                print(f"[score] chr{chrom}: {len(local_idx):,} variants "
                      f"({density*100:.1f}% of chr, {strategy})", flush=True)
        finally:
            reader.close()

    if prs_gpu is None:
        raise RuntimeError("no chromosomes scored — check cugen_dir")

    prs_np = cp.asnumpy(prs_gpu)
    if fid_order is not None:
        if len(fid_order) != n_samples:
            raise ValueError(
                f"fid_order length {len(fid_order)} != n_samples {n_samples}"
            )
        df = pd.DataFrame({"sample_id": list(fid_order), "prs": prs_np})
    else:
        df = pd.DataFrame({"sample_idx": np.arange(n_samples), "prs": prs_np})

    if output is not None:
        out = str(output)
        if out.endswith(".feather"):
            df.to_feather(out)
        else:
            sep = "\t" if out.endswith((".tsv", ".tsv.gz")) else ","
            compression = "gzip" if out.endswith(".gz") else None
            df.to_csv(out, sep=sep, index=False, compression=compression)
        if verbose:
            print(f"[score] wrote {out}", flush=True)

    if verbose:
        print(f"[score] done: {total_used:,} variants applied across "
              f"{len([s for s in per_chr_stats if s[1] > 0])} chrs",
              flush=True)
    return df
