"""cugen.assoc — Step 4: GWAS + orchestrated 1-3+4 ``ultralasso_gwas``.

Verbatim port of ``step4_gwas_multiprocess.py`` (production multi-process GWAS,
session 14 default = 8 workers + CUDA MPS). The single-process fallback path
in ``step4_gwas_array.py`` is collapsed into the same worker function used by
the pool (``_worker_gwas``) — calling ``gwas(..., n_workers=1)`` runs the
worker once in-process.

Production behaviour preserved:
  * Multi-process pool (spawn) + CUDA MPS; ``n_workers=8`` is current default.
  * Greedy chromosome bin-packing by variant count.
  * Wald test (production default) and score test (REGENIE-style, rare-variant
    flag) both available; betas byte-exact vs production.
  * Output sumstats columns: CHR, POS, ID, REF, ALT, BETA, SE, Z, P, MAF, gidx
    (mirrors production exactly — column names kept upper-case for back-compat
    with downstream LDSC / plotting code).

``ultralasso_gwas`` is the convenience orchestrator that wires
``screen_chromosome`` (×22) → ``fit_joint_lasso`` → ``gwas``. Lazy-imports
``screen`` and ``lasso`` so the bare ``gwas`` function remains import-safe even
while those modules are stubbed.

Env-var passthroughs (matches production):
  * ``USE_PINNED_READER=1``   — swap CugenReader → CugenReaderPinned via io.read_cugen
  * ``CUGEN_CHUNK_MB``        — pinned buffer size (default 64)
  * ``CUGEN_N_BUFFERS``       — pinned ring depth (default 2)
"""

from __future__ import annotations

import logging
import multiprocessing as mp
import os
import struct
import subprocess
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("cugen.assoc")


# ---------------------------------------------------------------------------
# Helpers — variant counts + chromosome distribution + lambda GC
# ---------------------------------------------------------------------------

def _get_variant_counts(cugen_dir: Union[str, Path]) -> dict:
    """Read n_variants from each chr*.cugen header (no CuPy needed)."""
    counts = {}
    cugen_dir = str(cugen_dir)
    for c in range(1, 23):
        path = os.path.join(cugen_dir, f"chr{c}.cugen")
        if os.path.exists(path):
            with open(path, 'rb') as f:
                f.seek(24)  # n_variants at offset 24
                n = struct.unpack('<Q', f.read(8))[0]
                counts[c] = n
    return counts


def _distribute_chromosomes(variant_counts: dict, n_workers: int):
    """Greedy bin-packing: assign largest-unassigned chr to least-loaded worker."""
    chrs_sorted = sorted(
        variant_counts.keys(),
        key=lambda c: variant_counts[c],
        reverse=True,
    )
    worker_loads = [0] * n_workers
    worker_chrs = [[] for _ in range(n_workers)]
    for c in chrs_sorted:
        min_idx = worker_loads.index(min(worker_loads))
        worker_chrs[min_idx].append(c)
        worker_loads[min_idx] += variant_counts[c]
    return worker_chrs, worker_loads


def _compute_lambda_gc(p_values: np.ndarray) -> float:
    """Genomic inflation factor (lambda GC).

    Uses the survival-function inverse ``chi2.isf(p)`` rather than
    ``chi2.ppf(1 - p)``: the latter forms ``1 - p`` which loses all precision
    for genome-wide-significant p (1 - 1e-50 == 1.0 in float64), pushing chi2
    to +inf and corrupting the median. isf evaluates the upper tail directly.
    """
    from scipy import stats as sp_stats
    p = np.asarray(p_values, dtype=np.float64)
    p = p[np.isfinite(p)]
    p = np.clip(p, 1e-300, 1.0)
    chi2 = sp_stats.chi2.isf(p, df=1)
    chi2 = chi2[np.isfinite(chi2)]
    if len(chi2) == 0:
        return float('nan')
    return float(np.median(chi2) / 0.4549364)


# ---------------------------------------------------------------------------
# Worker — process assigned chromosomes in its own CUDA context
# ---------------------------------------------------------------------------

def _worker_gwas(args_tuple):
    """
    Worker function: process assigned chromosomes.

    Called in a spawned process — imports CuPy fresh for clean CUDA context.
    Mirrors the production worker in ``step4_gwas_multiprocess.py`` byte-for-byte;
    only the reader import is changed (``from .io import …``) and the env-var
    swap to CugenReaderPinned is honoured.
    """
    (chromosomes, cugen_dir, loco_pred_path, annotation_path,
     output_dir, block_size, device, worker_id, test, maf_min) = args_tuple

    # Import CuPy here — each spawned process gets its own CUDA context.
    import cupy as cp
    from cupyx.scipy.special import erfc as gpu_erfc

    # Reader swap matches production: USE_PINNED_READER=1 -> CugenReaderPinned.
    use_pinned = os.environ.get('USE_PINNED_READER', '0') == '1'
    if use_pinned:
        from .io import CugenReaderPinned as _Reader
    else:
        from .io import CugenReader as _Reader

    cp.cuda.Device(device).use()

    # Load LOCO predictions (each worker loads independently — ~59 MB).
    loco_data = np.load(loco_pred_path, allow_pickle=True)
    loco_predictions = loco_data['predictions']
    y_original = loco_data['y_original']

    # Load annotation if provided.
    annotation_df = None
    if annotation_path and os.path.exists(annotation_path):
        annotation_df = pd.read_feather(annotation_path)

    worker_results = {}
    t_worker_start = time.time()

    for chr_num in sorted(chromosomes):
        cugen_path = os.path.join(cugen_dir, f"chr{chr_num}.cugen")
        if not os.path.exists(cugen_path):
            logger.warning("[W%d] %s not found, skipping", worker_id, cugen_path)
            continue

        t_chr = time.time()

        # LOCO residual.
        y_resid = y_original - loco_predictions[:, chr_num - 1]

        reader = _Reader(cugen_path, device=device)
        n_samples = reader.n_samples
        n_variants = reader.n_variants

        # Center the residual phenotype.
        y_mean = float(y_resid.mean())
        y_centered = y_resid - y_mean
        syy = float((y_centered ** 2).sum())

        y_gpu = cp.asarray(y_centered.astype(np.float32))

        mu_x, sxx, maf = reader.get_stats()
        gidx = reader.get_gidx()

        block_results = []
        for start in range(0, n_variants, block_size):
            end = min(start + block_size, n_variants)

            # Fused univariate regression (packed 2-bit -> beta directly).
            beta_gpu, num_gpu, den_gpu = reader.fused_univariate(
                y_gpu, start, end)

            sxx_block = cp.asarray(sxx[start:end])
            maf_block = maf[start:end]
            gidx_block = gidx[start:end]

            # SE / Z / P entirely on GPU (no cp.asnumpy in hot loop).
            SQRT2 = 1.4142135623730951
            MIN_DEN = 1e-20
            sxx64 = sxx_block.astype(cp.float64)
            den64 = cp.maximum(sxx64, MIN_DEN)
            beta64 = beta_gpu.astype(cp.float64)
            if test == "score":
                # REGENIE-style score test: global null variance syy/(n-1).
                # At MAC≈1 this stays bounded where per-variant RSS collapses.
                # Note (session 13): empirically near-identical to Wald on our
                # imputed data because rare-variant β̂ blowup is in the
                # ESTIMATE (X'y / X'X), not the SE. Flag kept for completeness.
                se64 = cp.sqrt(
                    cp.float64(syy)
                    / (cp.float64(n_samples - 1) * den64)
                )
            else:
                # Wald (default, production): per-variant residual variance.
                rss64 = cp.maximum(
                    cp.float64(syy) - beta64 ** 2 * sxx64, MIN_DEN
                )
                se64 = cp.sqrt(rss64 / (cp.float64(n_samples - 2) * den64))
            z64 = beta64 / cp.maximum(se64, MIN_DEN)
            p64 = gpu_erfc(cp.abs(z64) / SQRT2)

            # Single GPU→CPU transfer per block.
            block_df = pd.DataFrame({
                'gidx': gidx_block,
                'BETA': cp.asnumpy(beta_gpu),
                'SE': cp.asnumpy(se64).astype(np.float32),
                'Z': cp.asnumpy(z64).astype(np.float32),
                'P': cp.asnumpy(p64),
                'MAF': maf_block,
            })
            block_results.append(block_df)

        reader.close()
        gwas_df = pd.concat(block_results, ignore_index=True)

        # MAF filter at output stage (production default 1e-4).
        if maf_min is not None and maf_min > 0:
            gwas_df = gwas_df[gwas_df['MAF'] >= maf_min].reset_index(drop=True)

        # Merge with annotation.
        if annotation_df is not None:
            gwas_df = gwas_df.merge(annotation_df, on='gidx', how='left')
        else:
            gwas_df['CHR'] = chr_num
            gwas_df['POS'] = 0
            gwas_df['ID'] = '.'
            gwas_df['REF'] = '.'
            gwas_df['ALT'] = '.'

        # Standard column order.
        cols = ['CHR', 'POS', 'ID', 'REF', 'ALT', 'BETA', 'SE', 'Z', 'P',
                'MAF', 'gidx']
        out_cols = [c for c in cols if c in gwas_df.columns]
        gwas_df = gwas_df[out_cols]

        # Per-chr file (always written when output_dir is given — production
        # behaviour; the merge step pipes them together with awk/pigz).
        if output_dir is not None:
            out_path = os.path.join(output_dir, f"chr{chr_num}_sumstats.tsv")
            gwas_df.to_csv(out_path, sep='\t', index=False, float_format='%.6g')

        n_sig = int((gwas_df['P'] < 5e-8).sum())
        elapsed = time.time() - t_chr
        logger.info(
            "[W%d] chr%d: %d variants, %d sig, %.1fs",
            worker_id, chr_num, n_variants, n_sig, elapsed,
        )

        worker_results[chr_num] = gwas_df if output_dir is None else len(gwas_df)

        # Free GPU memory between chromosomes.
        del y_gpu, gwas_df
        cp.get_default_memory_pool().free_all_blocks()

    t_total = time.time() - t_worker_start
    if output_dir is None:
        total_vars = sum(len(df) for df in worker_results.values())
    else:
        total_vars = sum(worker_results.values())
    logger.info(
        "[W%d] Done: %d chr, %d variants, %.1fs",
        worker_id, len(worker_results), total_vars, t_total,
    )

    return worker_id, worker_results


# ---------------------------------------------------------------------------
# Worker — binary trait: logistic score test + selective SPA
# ---------------------------------------------------------------------------

# Clip applied before the logit bridge eta = logit(clip(mu_LPM)). Kept as a
# module constant so the legacy-parity adapter (which builds eta_loco the same
# way for step4_gwas_array_binary.py) uses an identical value → byte-exact match.
LOGIT_CLIP_EPS = 1e-6


def _worker_gwas_binary(args_tuple):
    """
    Worker function for binary-trait GWAS (logistic score test + selective SPA).

    Mirrors ``step4_gwas_array_binary.gwas_binary_chromosome`` exactly so the
    package path reproduces the validated legacy binary GWAS byte-for-byte.

    The null model is supplied entirely by a per-chromosome logit-scale offset
    ``eta``. The LOCO NPZ is accepted in two forms:
      * ``eta_loco`` (n, 22) + ``y_binary``  — logit-scale (step3_loco_binary),
        used directly.
      * ``predictions`` (n, 22) + ``y_original`` — LPM probability-scale (the
        continuous Fix-I LOCO from the pheno-wave). Bridged to logit via
        ``eta = logit(clip(mu_LPM, EPS, 1-EPS))`` so ``sigmoid(eta) = mu_LPM``
        and the logistic score collapses to the LPM-residual score U = X'(y-mu).
        This matches the manuscript Methods (LPM residual + score test + SPA).
    """
    (chromosomes, cugen_dir, loco_pred_path, annotation_path,
     output_dir, block_size, device, worker_id,
     min_mac, spa_pthresh, spa_sd_thresh, spa_chunk) = args_tuple

    import cupy as cp
    from scipy import stats as sp_stats
    from ._binary_spa import (
        apply_fast_spa_gpu,
        compute_binary_score_stats_gpu,
    )

    use_pinned = os.environ.get('USE_PINNED_READER', '0') == '1'
    if use_pinned:
        from .io import CugenReaderPinned as _Reader
    else:
        from .io import CugenReader as _Reader

    cp.cuda.Device(device).use()

    def _z_from_p_signed(p, sign_source):
        out = np.full_like(p, np.nan, dtype=np.float64)
        mask = np.isfinite(p)
        if np.any(mask):
            chi2 = sp_stats.chi2.isf(np.clip(p[mask], 1e-300, 1.0), df=1)
            out[mask] = np.sign(sign_source[mask]) * np.sqrt(np.maximum(chi2, 0.0))
        return out

    # Load LOCO offsets (each worker loads independently). Detect form + bridge.
    loco_data = np.load(loco_pred_path, allow_pickle=True)
    if 'eta_loco' in loco_data.files:
        eta_all = loco_data['eta_loco'].astype(np.float32)        # logit scale
        y = loco_data['y_binary'].astype(np.float32)
    else:
        mu_lpm = loco_data['predictions'].astype(np.float64)      # LPM prob scale
        mu_lpm = np.clip(mu_lpm, LOGIT_CLIP_EPS, 1.0 - LOGIT_CLIP_EPS)
        eta_all = np.log(mu_lpm / (1.0 - mu_lpm)).astype(np.float32)
        y = loco_data['y_original'].astype(np.float32)

    annotation_df = None
    if annotation_path and os.path.exists(annotation_path):
        annotation_df = pd.read_feather(annotation_path)

    worker_results = {}
    t_worker_start = time.time()

    y_gpu = cp.asarray(y, dtype=cp.float32)

    for chr_num in sorted(chromosomes):
        cugen_path = os.path.join(cugen_dir, f"chr{chr_num}.cugen")
        if not os.path.exists(cugen_path):
            logger.warning("[W%d] %s not found, skipping", worker_id, cugen_path)
            continue

        t_chr = time.time()
        eta_gpu = cp.asarray(eta_all[:, chr_num - 1], dtype=cp.float32)

        reader = _Reader(cugen_path, device=device)
        n_samples = int(reader.n_samples)
        n_variants = int(reader.n_variants)

        mu_x, sxx, maf = reader.get_stats()
        gidx = reader.get_gidx()
        mac_all = (2.0 * maf * n_samples).astype(np.float32)

        block_results = []
        for start in range(0, n_variants, block_size):
            end = min(start + block_size, n_variants)
            X_gpu = reader.read_to_gpu(start, end)
            score_stats = compute_binary_score_stats_gpu(X_gpu, y_gpu, eta_gpu)

            mac_block = mac_all[start:end]
            valid_mask_np = mac_block >= min_mac
            valid_mask_gpu = cp.asarray(valid_mask_np)

            p_norm_gpu = score_stats["p_norm"]
            spa_mask_gpu = (
                valid_mask_gpu
                & (p_norm_gpu < spa_pthresh)
                & (cp.abs(score_stats["z"]) > spa_sd_thresh)
            )

            p_final_gpu = apply_fast_spa_gpu(
                X_gpu, score_stats, spa_mask_gpu, spa_chunk=spa_chunk)

            beta_np = cp.asnumpy(score_stats["beta"]).astype(np.float64)
            se_np = cp.asnumpy(score_stats["se"]).astype(np.float64)
            z_norm_np = cp.asnumpy(score_stats["z"]).astype(np.float64)
            p_norm_np = cp.asnumpy(p_norm_gpu).astype(np.float64)
            p_final_np = cp.asnumpy(p_final_gpu).astype(np.float64)
            spa_mask_np = cp.asnumpy(spa_mask_gpu)

            p_final_np[~valid_mask_np] = np.nan
            z_final_np = _z_from_p_signed(p_final_np, z_norm_np)

            method = np.where(
                ~valid_mask_np,
                "LOW_MAC",
                np.where(spa_mask_np, "SPA", "SCORE"),
            )

            block_results.append(pd.DataFrame({
                'gidx': gidx[start:end],
                'BETA': beta_np,
                'SE': se_np,
                'Z': z_final_np,
                'P': p_final_np,
                'P_NORM': p_norm_np,
                'TEST': method,
                'MAF': maf[start:end],
                'MAC': mac_block,
            }))

            del X_gpu, score_stats, p_final_gpu, valid_mask_gpu, spa_mask_gpu
            cp.get_default_memory_pool().free_all_blocks()

        reader.close()
        gwas_df = pd.concat(block_results, ignore_index=True)

        if annotation_df is not None:
            gwas_df = gwas_df.merge(annotation_df, on='gidx', how='left')
        else:
            gwas_df['CHR'] = chr_num
            gwas_df['POS'] = 0
            gwas_df['ID'] = '.'
            gwas_df['REF'] = '.'
            gwas_df['ALT'] = '.'

        cols = ['CHR', 'POS', 'ID', 'REF', 'ALT', 'BETA', 'SE', 'Z', 'P',
                'P_NORM', 'TEST', 'MAF', 'MAC', 'gidx']
        out_cols = [c for c in cols if c in gwas_df.columns]
        gwas_df = gwas_df[out_cols]

        if output_dir is not None:
            out_path = os.path.join(output_dir, f"chr{chr_num}_sumstats.tsv")
            gwas_df.to_csv(out_path, sep='\t', index=False, float_format='%.6g')

        n_sig = int(np.nansum(gwas_df['P'].to_numpy() < 5e-8))
        logger.info(
            "[W%d] chr%d (binary): %d variants, %d sig, %.1fs",
            worker_id, chr_num, n_variants, n_sig, time.time() - t_chr,
        )

        worker_results[chr_num] = gwas_df if output_dir is None else len(gwas_df)
        del gwas_df
        cp.get_default_memory_pool().free_all_blocks()

    t_total = time.time() - t_worker_start
    if output_dir is None:
        total_vars = sum(len(df) for df in worker_results.values())
    else:
        total_vars = sum(worker_results.values())
    logger.info(
        "[W%d] Done (binary): %d chr, %d variants, %.1fs",
        worker_id, len(worker_results), total_vars, t_total,
    )

    return worker_id, worker_results


# ---------------------------------------------------------------------------
# Merge — pigz-based concatenation of per-chr TSVs to combined gzip
# ---------------------------------------------------------------------------

def _merge_results(output_dir: Union[str, Path], chromosomes: Sequence[int]):
    """Merge per-chromosome TSV files into combined gzipped file (awk + pigz)."""
    output_dir = str(output_dir)
    combined_path = os.path.join(output_dir, "all_sumstats.tsv.gz")
    files = []
    for c in sorted(chromosomes):
        path = os.path.join(output_dir, f"chr{c}_sumstats.tsv")
        if os.path.exists(path):
            files.append(path)
    if not files:
        logger.warning("No chromosome result files found!")
        return None, []

    cmd = (
        f"awk 'FNR==1 && NR!=1{{next}} 1' "
        f"{' '.join(files)} | pigz -p 4 > {combined_path}"
    )
    subprocess.run(cmd, shell=True, check=True)

    result = subprocess.run(
        f"pigz -dc {combined_path} | wc -l",
        shell=True, capture_output=True, text=True,
    )
    total_lines = int(result.stdout.strip()) - 1  # subtract header

    logger.info("Saved combined results: %s (%d variants)", combined_path, total_lines)
    return total_lines, files


# ---------------------------------------------------------------------------
# Public API — gwas + ultralasso_gwas
# ---------------------------------------------------------------------------

def gwas(
    cugen_dir: Union[str, Path],
    cohort_npz: Union[str, Path],
    loco_predictions,
    *,
    n_workers: int = 8,
    family: str = "linear",
    test: str = "wald",
    maf_min: float = 1e-4,
    output: Optional[Union[str, Path]] = None,
    block_size: int = 4096,
    device: int = 0,
    annotation: Optional[Union[str, Path]] = None,
    min_mac: float = 20.0,
    spa_pthresh: float = 0.05,
    spa_sd_thresh: float = 2.0,
    spa_chunk: int = 64,
) -> pd.DataFrame:
    """Run Step-4 univariate GWAS over all variants in ``cugen_dir``.

    Verbatim production port of ``step4_gwas_multiprocess.py``. Multi-process
    pool (spawn) + CUDA MPS — session-14 default is 8 workers on 16 CPUs.

    Parameters
    ----------
    cugen_dir : directory containing chr{1..22}.cugen (array OR imputed).
    cohort_npz : residualised cohort NPZ (output of prepare_cohort).
        Currently unused in the per-chromosome compute kernel (Wald uses the
        LOCO-residual phenotype directly), but kept in the public signature so
        downstream extensions (binary direct fit, score test with covariates)
        have a canonical handle to the cohort metadata.
    loco_predictions : np.ndarray (n_samples, 22) or path to step3 .npz.
        If an array is passed, it must be paired with the cohort's
        ``y_original`` — wrap in a dict-like .npz before passing, or pass the
        path directly (preferred).
    n_workers : multiprocess workers (default 8 — session-14 sweet spot,
        requires CUDA MPS + --cpus-per-task=16). ``n_workers=1`` runs the
        worker in-process without spawning a pool.
    family : 'linear' (continuous Wald/score; default) or 'binary'
        ('logistic' is an accepted alias). The binary path runs the SAIGE-style
        logistic score test + selective saddlepoint (SPA) correction
        (``_worker_gwas_binary`` / ``_binary_spa``): the per-chromosome LOCO
        offset is the null model, MAC < ``min_mac`` is masked LOW_MAC, and SPA
        is applied where the normal tail is suspect
        (``p_norm < spa_pthresh & |Z| > spa_sd_thresh``). The LOCO NPZ may carry
        ``eta_loco``/``y_binary`` (logit scale, from step3_loco_binary) or
        ``predictions``/``y_original`` (LPM probability scale — bridged via
        ``eta = logit(clip(mu_LPM))``, matching the manuscript LPM-residual
        Methods). Reproduces ``step4_gwas_array_binary.py`` byte-for-byte.
    test : 'wald' (production) or 'score' (REGENIE-style, MAC-robust in theory
        but empirically near-identical to Wald on imputed biobank — session 13).
        Ignored when ``family='binary'`` (always score + selective SPA).
    maf_min : MAF filter applied to the output (default 1e-4). Linear only;
        the binary path uses ``min_mac`` masking instead (no MAF drop).
    min_mac, spa_pthresh, spa_sd_thresh, spa_chunk : binary-path SPA controls
        (defaults 20 / 0.05 / 2.0 / 64). ``min_mac`` is the standard biobank
        post-hoc minor-allele-count floor below which a variant is LOW_MAC.
    output : sumstats tsv.gz path; if None, returns DataFrame in memory.
        If a directory is given (or output ends with `/`), per-chromosome TSVs
        plus all_sumstats.tsv.gz are written there (production layout).
    block_size : variants/block (default 4096).
    device : GPU device id (default 0).
    annotation : optional path to gidx_annotation feather. If omitted, the
        output has only placeholder CHR/POS/ID/REF/ALT columns.

    Returns
    -------
    pandas.DataFrame
        Sumstats. Columns (production order):
        CHR, POS, ID, REF, ALT, BETA, SE, Z, P, MAF, gidx.
    """
    # 'logistic' is an alias for the binary score+SPA path.
    if family == "logistic":
        family = "binary"
    if family not in ("linear", "binary"):
        raise ValueError(
            f"gwas(family={family!r}): expected 'linear' or 'binary'.")
    if family == "linear" and test not in ("wald", "score"):
        raise ValueError(f"gwas(test={test!r}): expected 'wald' or 'score'.")

    cugen_dir = str(cugen_dir)

    # `loco_predictions` may be an in-memory ndarray or a path. The worker
    # signature expects a path (each worker `np.load`s independently) so an
    # in-memory ndarray + cohort_npz tuple is materialised to a temp .npz.
    if isinstance(loco_predictions, (str, Path)):
        loco_pred_path = str(loco_predictions)
    else:
        if cohort_npz is None:
            raise ValueError(
                "gwas: when passing loco_predictions as an array, cohort_npz "
                "must be supplied so the worker can fetch y_original."
            )
        cohort = np.load(str(cohort_npz), allow_pickle=True)
        import tempfile
        tmp = tempfile.NamedTemporaryFile(
            suffix="_loco_pred.npz", delete=False)
        tmp.close()
        np.savez(
            tmp.name,
            predictions=np.asarray(loco_predictions),
            y_original=cohort['y_original'],
        )
        loco_pred_path = tmp.name

    # Resolve output_dir + decide whether to return a DataFrame or write files.
    if output is None:
        output_dir = None
        return_df = True
    else:
        output_p = Path(output)
        if output_p.suffix == "" or str(output).endswith("/"):
            output_dir = str(output_p)
            os.makedirs(output_dir, exist_ok=True)
            return_df = False
        else:
            # output is a file path — write per-chr to its parent + concatenate.
            output_dir = str(output_p.parent)
            os.makedirs(output_dir, exist_ok=True)
            return_df = False

    # Read variant counts and distribute chromosomes.
    variant_counts = _get_variant_counts(cugen_dir)
    if not variant_counts:
        raise FileNotFoundError(f"No chr*.cugen files in {cugen_dir!r}")

    n_workers = max(1, min(int(n_workers), len(variant_counts)))
    worker_chrs, worker_loads = _distribute_chromosomes(variant_counts, n_workers)

    logger.info(
        "GWAS: %d chromosomes, %d total variants, %d workers",
        len(variant_counts), sum(variant_counts.values()), n_workers,
    )
    for i, (chrs, load) in enumerate(zip(worker_chrs, worker_loads)):
        chr_str = ','.join(str(c) for c in sorted(chrs))
        logger.info("  Worker %d: chr[%s] = %d variants", i, chr_str, load)

    # When the caller wants a DataFrame in memory, force the worker not to
    # write per-chr files (pass output_dir=None to the worker) and collect the
    # frames it returns. Multi-process with in-memory return is supported but
    # pickling 6.8M-row DataFrames is wasteful — the production code always
    # writes to disk, and we follow that pattern.
    worker_output_dir = output_dir if not return_df else None

    worker_fn = _worker_gwas_binary if family == "binary" else _worker_gwas
    worker_args = []
    for i, chrs in enumerate(worker_chrs):
        if family == "binary":
            worker_args.append((
                sorted(chrs),
                cugen_dir,
                loco_pred_path,
                str(annotation) if annotation else None,
                worker_output_dir,
                block_size,
                device,
                i,
                min_mac,
                spa_pthresh,
                spa_sd_thresh,
                spa_chunk,
            ))
        else:
            worker_args.append((
                sorted(chrs),
                cugen_dir,
                loco_pred_path,
                str(annotation) if annotation else None,
                worker_output_dir,
                block_size,
                device,
                i,
                test,
                maf_min,
            ))

    t_start = time.time()
    if n_workers == 1:
        # In-process — keeps the call import-safe under multiprocessing-disabled
        # contexts (e.g. running inside a notebook on a login node smoke test).
        results = [worker_fn(worker_args[0])]
    else:
        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            results = pool.map(worker_fn, worker_args)
    t_compute = time.time() - t_start
    logger.info("All workers completed in %.1fs", t_compute)

    # Merge / return.
    if return_df:
        # Worker returned DataFrames in worker_results.
        all_dfs = []
        for _, worker_result in results:
            for chr_num in sorted(worker_result.keys()):
                df = worker_result[chr_num]
                # In multi-worker mode the worker would have written to disk
                # if output_dir was set; in in-memory mode it returns DataFrames.
                if isinstance(df, pd.DataFrame):
                    all_dfs.append(df)
        combined = pd.concat(all_dfs, ignore_index=True) if all_dfs else pd.DataFrame()
        return combined

    # Disk-output path — merge per-chr TSVs.
    all_chromosomes = sorted(variant_counts.keys())
    _merge_results(output_dir, all_chromosomes)

    # If a specific file path was requested (not just a dir), rename
    # all_sumstats.tsv.gz to that file.
    if output is not None:
        output_p = Path(output)
        if output_p.suffix in (".gz",) or (output_p.suffix == ".tsv" and not output_p.is_dir()):
            default = Path(output_dir) / "all_sumstats.tsv.gz"
            if default.resolve() != output_p.resolve() and default.exists():
                os.replace(default, output_p)

    # For disk-output callers, also return a (lazy-loaded) DataFrame for
    # back-compat with code that does `df = gwas(...)`.
    final = Path(output_dir) / "all_sumstats.tsv.gz"
    if final.exists():
        return pd.read_csv(final, sep='\t', compression='gzip')
    return pd.DataFrame()


def ultralasso_gwas(
    cohort_npz: Union[str, Path],
    cugen_dir: Union[str, Path],
    *,
    cugen_dir_step4: Optional[Union[str, Path]] = None,
    covariates: Optional[Sequence[str]] = None,
    alpha_screen: float = 6e-4,
    alpha_joint: float = 6e-3,
    ridge: float = 1e-2,
    loco_mode: str = "ols",
    n_workers_step4: int = 8,
    output_dir: Optional[Union[str, Path]] = None,
) -> dict:
    """Run the full UltraLasso GWAS pipeline (Steps 1+2 → 3 → 4) in one call.

    Orchestrator: ``screen_chromosome`` (×22) → ``fit_joint_lasso`` → ``gwas``.
    Defaults match the production pipeline (see CLAUDE.md):

      * covariates = ``['age', 'sex', 'PC1', ..., 'PC10']``
      * loco_mode  = ``'ols'``
      * ridge      = ``1e-2``
      * alpha_screen / alpha_joint = ``6e-4`` / ``6e-3`` (height; T2D B0 uses
        ``1e-5`` / ``1e-4`` — pass explicitly)
      * n_workers_step4 = ``8`` (session-14 sweet spot)

    Lazy-imports screen + lasso so a bare ``gwas`` import never trips on
    those modules being stubbed/in-progress.

    Returns
    -------
    dict
        {'screen': pd.DataFrame, 'joint_lasso': dict, 'sumstats': pd.DataFrame}
    """
    # Lazy imports — keeps gwas() useable while screen/lasso are still stubs.
    from .screen import screen_chromosome
    from .lasso import fit_joint_lasso

    if covariates is None:
        covariates = ['age', 'sex'] + [f'Global_PC{i}' for i in range(1, 11)]

    if output_dir is not None:
        output_dir = str(output_dir)
        os.makedirs(output_dir, exist_ok=True)

    cugen_dir = str(cugen_dir)
    cugen_dir_step4 = str(cugen_dir_step4) if cugen_dir_step4 else cugen_dir

    # Step 1+2: per-chromosome screening.
    logger.info("ultralasso_gwas: Step 1+2 screening (alpha=%g)", alpha_screen)
    screen_dfs = []
    for chrom in range(1, 23):
        cugen_path = os.path.join(cugen_dir, f"chr{chrom}.cugen")
        if not os.path.exists(cugen_path):
            logger.warning("chr%d.cugen missing — skipping screen", chrom)
            continue
        df = screen_chromosome(
            chrom=chrom,
            cohort_npz=cohort_npz,
            cugen_path=cugen_path,
            alpha=alpha_screen,
            output=(os.path.join(output_dir, f"step1_to_step2_chr{chrom}.feather")
                    if output_dir else None),
        )
        screen_dfs.append(df)
    candidates = pd.concat(screen_dfs, ignore_index=True) if screen_dfs else pd.DataFrame()
    logger.info("  selected candidates: %d", len(candidates))

    # Step 3: joint LASSO + LOCO predictions.
    logger.info("ultralasso_gwas: Step 3 joint LASSO (alpha=%g, ridge=%g, loco_mode=%s)",
                alpha_joint, ridge, loco_mode)
    step3 = fit_joint_lasso(
        candidates=candidates,
        cohort_npz=cohort_npz,
        cugen_dir=cugen_dir,
        covariates=covariates,
        alpha=alpha_joint,
        ridge=ridge,
        loco_mode=loco_mode,
        output_dir=output_dir,
    )

    # Step 4: GWAS using LOCO predictions.
    logger.info("ultralasso_gwas: Step 4 GWAS (n_workers=%d)", n_workers_step4)
    sumstats = gwas(
        cugen_dir=cugen_dir_step4,
        cohort_npz=cohort_npz,
        loco_predictions=step3['loco_predictions'],
        n_workers=n_workers_step4,
        family='linear',
        test='wald',
        output=(os.path.join(output_dir, 'gwas') if output_dir else None),
    )

    return {
        'screen': candidates,
        'joint_lasso': step3,
        'sumstats': sumstats,
    }
