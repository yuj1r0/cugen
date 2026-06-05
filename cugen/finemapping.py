"""cugen.finemapping — Step 5b: UltraSuSiE (Tier 1) + UltraMAP (Tier 2).

v0.1.3 (session 34): self-contained Python-API wrapper for the Step 5b
fine-mapping engine (UltraSuSiE Tier 1 + UltraMAP Tier 2). The heavy
implementation lives at :mod:`cugen._step5b_finemapping` (vendored
in-package — no sys.path tricks, no dependence on the parent unilassoGPU
repo). This module:

1.  Imports the in-package helpers ``_worker_finemap``,
    ``_process_chromosome_loci``, ``_bin_pack_chromosomes`` from
    :mod:`cugen._step5b_finemapping`.
2.  Exposes two clean entry points — :func:`ultrasusie` (Tier 1) and
    :func:`ultramap` (Tier 2) — that accept paths *or* in-memory objects
    (loci as ``pd.DataFrame`` or TSV path; LOCO predictions as ``np.ndarray``
    or NPZ path), build the cfg dict, spawn the multiprocess Pool with
    chromosome bin-packing, and assemble combined TSVs.

Production behaviour preserved::
  * Tier 1 (UltraSuSiE): SuSiE-RSS only, genome-wide; multiprocess via
    Pool(spawn) with greedy locus-count bin-packing.
  * Tier 2 (UltraMAP): dual-engine SuSiE + LASSO CPSS with consensus pPIPs.
  * Adaptive weights: ``adaptive_weight_method ∈ {'z', 'strength'}``.
  * LASSO credible-set level: ``lasso_cs_level ∈ {50..90}``.
  * Pinned-reader path is honoured via the in-package :class:`cugen.io.CugenReader`
    + ``USE_PINNED_READER=1`` env flag.

Output (one row per locus-variant pair)::

    all_variants.tsv.gz   gzipped per-variant detail
    loci_summary.tsv      one row per locus, columns differ by tier

The wrappers also return a dict::

    {
        'per_variant': DataFrame,   # all_variants.tsv.gz as DataFrame
        'summary':     DataFrame,   # loci_summary.tsv as DataFrame
        'output_dir':  Path,
        'wall_s':      float,
    }
"""

from __future__ import annotations

import multiprocessing as mp
import os
import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


# Lazy import — heavy GPU deps (cupy) are only needed at call time, not at
# package import on a CPU-only login node.
_step5b = None


def _load_step5b():
    """Import the vendored step5b module on first call."""
    global _step5b
    if _step5b is None:
        from . import _step5b_finemapping as _mod  # noqa: PLC0415
        _step5b = _mod
    return _step5b


# ----------------------------------------------------------------------------
# Input resolution helpers (accept path or in-memory object).
# ----------------------------------------------------------------------------
def _resolve_loci(loci) -> pd.DataFrame:
    """Accept DataFrame or TSV path; return DataFrame.

    Required columns: ``locus_id``, ``CHR``, ``start_bp``, ``end_bp``.
    """
    if isinstance(loci, pd.DataFrame):
        return loci.copy()
    if isinstance(loci, (str, Path)):
        return pd.read_csv(str(loci), sep="\t")
    raise TypeError(f"loci must be DataFrame or path; got {type(loci)}")


def _resolve_loco(loco_predictions, expect_keys=("predictions", "y_original")):
    """Accept dict-of-arrays, ``(predictions, y_original)`` tuple, NPZ path, or ndarray.

    Returns ``(predictions, y_original, source_path_or_none)``. The source path
    is needed by the worker — it loads its own NPZ copy to stay spawn-safe.
    """
    if isinstance(loco_predictions, (str, Path)):
        loaded = np.load(str(loco_predictions), allow_pickle=True)
        return (
            loaded["predictions"],
            loaded["y_original"],
            str(loco_predictions),
        )
    if isinstance(loco_predictions, dict):
        return (
            loco_predictions["predictions"],
            loco_predictions["y_original"],
            loco_predictions.get("_npz_path"),
        )
    raise TypeError(
        "loco_predictions must be an NPZ path or dict with 'predictions' + "
        "'y_original' keys"
    )


def _resolve_annotation(annotation) -> tuple[pd.DataFrame, Optional[str]]:
    """Accept DataFrame or feather path; return (DataFrame, source_path_or_none).

    Workers need the path so they can reload — passing a DataFrame through
    spawn-pickle is wasteful and not stable across pandas versions.
    """
    if isinstance(annotation, pd.DataFrame):
        df = annotation.copy()
        if "CHR" in df.columns:
            df["CHR"] = df["CHR"].astype(str)
        return df, None
    if isinstance(annotation, (str, Path)):
        df = pd.read_feather(str(annotation))
        df["CHR"] = df["CHR"].astype(str)
        return df, str(annotation)
    raise TypeError(f"annotation must be DataFrame or feather path; got {type(annotation)}")


# ----------------------------------------------------------------------------
# Core dispatcher — runs either tier, sequential or multi-worker.
# ----------------------------------------------------------------------------
def _run_finemap(
    *,
    tier: int,
    loci,
    loco_predictions,
    cohort_npz: Union[str, Path],  # for prepare_cohort consistency; not used directly
    cugen_dir: Union[str, Path],
    annotation: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    gwas_sumstats_dir: Optional[Union[str, Path]] = None,
    n_signals: int = 10,
    n_workers: int = 1,
    coverage: float = 0.95,  # noqa: ARG001 — placeholder; SuSiE uses internal 0.95
    n_pairs: int = 50,
    n_lambda: int = 20,
    ridge_alpha: float = 1e-4,
    max_variants: int = 15000,
    device: int = 0,
    low_memory: bool = False,
    batch_size: int = 8192,
    enable_convergence: bool = False,
    max_rounds: int = 3,
    discrepancy_threshold: float = 0.3,
    prior_strength: float = 0.5,
    cache_stats: bool = False,
    load_stats: bool = False,
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_clip_min: float = 0.2,
    adaptive_weight_clip_max: float = 5.0,
    lambda_min_ratio: float = 0.01,
    adaptive_weight_method: str = "z",
    lasso_cs_level: int = 60,
    coding_gidx_set: Optional[set] = None,
    coding_prior_bonus: float = 1.0,
    verbose: bool = True,
) -> dict:
    s5b = _load_step5b()
    import cupy as cp  # noqa: PLC0415 — gated on GPU availability at call

    cp.cuda.Device(device).use()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    loci_df = _resolve_loci(loci)
    loci_df = loci_df.copy()
    loci_df["CHR"] = loci_df["CHR"].astype(int)
    if len(loci_df) == 0:
        return {
            "per_variant": pd.DataFrame(),
            "summary": pd.DataFrame(),
            "output_dir": output_dir,
            "wall_s": 0.0,
        }

    loco_pred_arr, y_original, loco_pred_path = _resolve_loco(loco_predictions)
    annotation_df, annotation_path = _resolve_annotation(annotation)

    chrs_sorted = sorted(loci_df["CHR"].unique())
    loci_by_chr = {
        int(c): loci_df[loci_df["CHR"] == c].copy() for c in chrs_sorted
    }

    cfg = {
        "cugen_dir": str(cugen_dir),
        "output_dir": str(output_dir),
        "gwas_sumstats_dir": (str(gwas_sumstats_dir) if gwas_sumstats_dir
                              is not None else None),
        "n_pairs": n_pairs,
        "n_lambda": n_lambda,
        "ridge_alpha": ridge_alpha,
        "susie_L": n_signals,
        "max_variants": max_variants,
        "device": device,
        "low_memory": low_memory,
        "batch_size": batch_size,
        "enable_convergence": enable_convergence,
        "max_rounds": max_rounds,
        "discrepancy_threshold": discrepancy_threshold,
        "prior_strength": prior_strength,
        "tier": tier,
        "cache_stats": cache_stats,
        "load_stats": load_stats,
        "adaptive_weight_gamma": adaptive_weight_gamma,
        "adaptive_weight_clip_min": adaptive_weight_clip_min,
        "adaptive_weight_clip_max": adaptive_weight_clip_max,
        "lambda_min_ratio": lambda_min_ratio,
        "adaptive_weight_method": adaptive_weight_method,
        "lasso_cs_level": lasso_cs_level,
        "coding_prior_bonus": coding_prior_bonus,
    }

    all_results = []
    loci_summary = []
    wall_t0 = time.time()

    if n_workers <= 1 or len(chrs_sorted) <= 1:
        # Sequential path — same as step5b_finemapping main().
        for chr_num in chrs_sorted:
            res, summ = s5b._process_chromosome_loci(
                int(chr_num), loci_by_chr[int(chr_num)], cfg, coding_gidx_set,
                loco_pred_arr, y_original, annotation_df,
            )
            all_results.extend(res)
            loci_summary.extend(summ)
    else:
        # Multi-worker path. Workers reload loco predictions + annotation
        # from disk; pass paths through the payload (spawn-safe).
        if loco_pred_path is None:
            raise ValueError(
                "Multi-worker fine-mapping requires loco_predictions to be a "
                "path (so each spawned worker can reload it). Pass the NPZ "
                "path instead of the in-memory dict."
            )
        if annotation_path is None:
            raise ValueError(
                "Multi-worker fine-mapping requires annotation to be a path "
                "(feather). Pass the file path instead of the in-memory "
                "DataFrame."
            )

        chr_locus_counts = {int(c): len(loci_by_chr[int(c)]) for c in chrs_sorted}
        actual_workers = min(int(n_workers), len(chrs_sorted))
        worker_chrs, worker_loads = s5b._bin_pack_chromosomes(
            chr_locus_counts, actual_workers,
        )
        if verbose:
            print(f"\nMulti-worker dispatch: {actual_workers} workers "
                  f"(requested {n_workers})", flush=True)
            for wi, (ws, ld) in enumerate(zip(worker_chrs, worker_loads)):
                print(f"  [W{wi}] chr{ws} - {ld} loci", flush=True)

        coding_gidx_list = (
            None if coding_gidx_set is None else list(coding_gidx_set)
        )

        payloads = []
        for wi, chrs_for_w in enumerate(worker_chrs):
            if not chrs_for_w:
                continue
            cfg_w = dict(cfg)
            cfg_w["_loci_by_chr"] = {c: loci_by_chr[c] for c in chrs_for_w}
            payloads.append((
                wi, chrs_for_w, cfg_w, coding_gidx_list,
                loco_pred_path, annotation_path,
            ))

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=len(payloads)) as pool:
            for (worker_results, worker_summary) in pool.imap_unordered(
                    s5b._worker_finemap, payloads):
                all_results.extend(worker_results)
                loci_summary.extend(worker_summary)

    wall_s = time.time() - wall_t0

    # Write outputs (mirrors step5b_finemapping main()).
    summary_df = pd.DataFrame(loci_summary) if loci_summary else pd.DataFrame()
    if len(summary_df):
        summary_path = output_dir / "loci_summary.tsv"
        summary_df.to_csv(summary_path, sep="\t", index=False)
        if verbose:
            print(f"Saved loci summary to {summary_path}", flush=True)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = output_dir / "all_variants.tsv.gz"
        combined.to_csv(combined_path, sep="\t", index=False, compression="gzip")
        if verbose:
            print(f"Saved {len(combined):,} variants to {combined_path}",
                  flush=True)
    else:
        combined = pd.DataFrame()

    if verbose:
        tier_name = "UltraSuSiE" if tier == 1 else "UltraMAP"
        print(f"\n{tier_name} (Tier {tier}) wall time: {wall_s:.1f}s",
              flush=True)

    return {
        "per_variant": combined,
        "summary": summary_df,
        "output_dir": output_dir,
        "wall_s": wall_s,
    }


# ----------------------------------------------------------------------------
# Public API.
# ----------------------------------------------------------------------------
def ultrasusie(
    loci,
    loco_predictions,
    cohort_npz: Union[str, Path],
    cugen_dir: Union[str, Path],
    *,
    annotation: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    n_signals: int = 10,
    n_workers: int = 8,
    coverage: float = 0.95,
    n_pairs: int = 50,
    n_lambda: int = 20,
    ridge_alpha: float = 1e-4,
    max_variants: int = 15000,
    device: int = 0,
    low_memory: bool = False,
    batch_size: int = 8192,
    cache_stats: bool = False,
    load_stats: bool = False,
    verbose: bool = True,
) -> dict:
    """Tier-1 genome-wide fine-mapping (SuSiE-RSS only).

    Parameters mirror the production CLI ``step5b_finemapping.py --tier 1``.
    See module docstring for output shape. ``n_workers >= 2`` requires a path
    for ``loco_predictions`` (NPZ) and ``annotation`` (feather) — workers
    reload from disk to stay spawn-safe.
    """
    return _run_finemap(
        tier=1,
        loci=loci,
        loco_predictions=loco_predictions,
        cohort_npz=cohort_npz,
        cugen_dir=cugen_dir,
        annotation=annotation,
        output_dir=output_dir,
        n_signals=n_signals,
        n_workers=n_workers,
        coverage=coverage,
        n_pairs=n_pairs,
        n_lambda=n_lambda,
        ridge_alpha=ridge_alpha,
        max_variants=max_variants,
        device=device,
        low_memory=low_memory,
        batch_size=batch_size,
        cache_stats=cache_stats,
        load_stats=load_stats,
        verbose=verbose,
    )


def ultramap(
    loci,
    loco_predictions,
    cohort_npz: Union[str, Path],
    cugen_dir: Union[str, Path],
    *,
    annotation: Union[str, Path, pd.DataFrame],
    output_dir: Union[str, Path],
    gwas_sumstats_dir: Optional[Union[str, Path]] = None,
    n_signals: int = 10,
    n_workers: int = 8,
    coverage: float = 0.95,
    n_pairs: int = 50,
    n_lambda: int = 20,
    ridge_alpha: float = 1e-4,
    max_variants: int = 15000,
    device: int = 0,
    low_memory: bool = False,
    batch_size: int = 8192,
    enable_convergence: bool = False,
    max_rounds: int = 3,
    discrepancy_threshold: float = 0.3,
    prior_strength: float = 0.5,
    adaptive_weight_method: str = "z",
    adaptive_weight_gamma: float = 1.0,
    adaptive_weight_clip_min: float = 0.2,
    adaptive_weight_clip_max: float = 5.0,
    lambda_min_ratio: float = 0.01,
    lasso_cs_level: int = 50,
    cache_stats: bool = False,
    load_stats: bool = False,
    coding_gidx_set: Optional[set] = None,
    coding_prior_bonus: float = 1.0,
    verbose: bool = True,
) -> dict:
    """Tier-2 dual-engine fine-mapping (SuSiE + LASSO CPSS with consensus pPIPs).

    Parameters mirror the production CLI ``step5b_finemapping.py --tier 2``.
    See module docstring for output shape.
    """
    return _run_finemap(
        tier=2,
        loci=loci,
        loco_predictions=loco_predictions,
        cohort_npz=cohort_npz,
        cugen_dir=cugen_dir,
        annotation=annotation,
        output_dir=output_dir,
        gwas_sumstats_dir=gwas_sumstats_dir,
        n_signals=n_signals,
        n_workers=n_workers,
        coverage=coverage,
        n_pairs=n_pairs,
        n_lambda=n_lambda,
        ridge_alpha=ridge_alpha,
        max_variants=max_variants,
        device=device,
        low_memory=low_memory,
        batch_size=batch_size,
        enable_convergence=enable_convergence,
        max_rounds=max_rounds,
        discrepancy_threshold=discrepancy_threshold,
        prior_strength=prior_strength,
        adaptive_weight_method=adaptive_weight_method,
        adaptive_weight_gamma=adaptive_weight_gamma,
        adaptive_weight_clip_min=adaptive_weight_clip_min,
        adaptive_weight_clip_max=adaptive_weight_clip_max,
        lambda_min_ratio=lambda_min_ratio,
        lasso_cs_level=lasso_cs_level,
        cache_stats=cache_stats,
        load_stats=load_stats,
        coding_gidx_set=coding_gidx_set,
        coding_prior_bonus=coding_prior_bonus,
        verbose=verbose,
    )
