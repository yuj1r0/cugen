"""cugen.lasso — Step 3: joint genome-wide LASSO + LOCO residuals.

Ported verbatim from ``step3_loco.py`` (production Fix-I default).

Production behaviour preserved:
  * FISTA solver (float32 X, accumulators promoted as needed)
  * Ridge λ = 1e-2 (Fix-I default) for LOCO OLS refit
  * LOCO mode: 'ols' (the production pipeline, exact X'X solve, ~14 s) or
    'lasso' (legacy v5 masked subtraction, ~1 s — script-internal name
    'masked'; this module accepts both spellings)
  * Covariates centred + scaled; intercept absorbed via y-mean centring
  * cp.linalg.solve + f64 cast + ridge=1e-2 (NOT Moore-Penrose pseudo-inverse)
  * Batched 2D@1D matmul to dodge CuPy's float32→float64 upcast OOM
  * 2048-feature batched LOCO predictions to dodge fancy-index OOM
  * Output: full_model.feather + loco_predictions.npz (22 cols)

Pinned reader auto-swap via ``USE_PINNED_READER=1`` mirrors production.
"""
import logging
import os
import time
from glob import glob
from pathlib import Path
from typing import Optional, Sequence, Union

import cupy as cp
import numpy as np
import pandas as pd
import polars as pl

from .io import CugenReader, CugenReaderPinned

logger = logging.getLogger("cugen.lasso")


# ---------------------------------------------------------------------------
# Reader selection mirrors step1/step2/step3 production behaviour.
# ---------------------------------------------------------------------------
def _reader_cls():
    if os.environ.get("USE_PINNED_READER", "0") == "1":
        return CugenReaderPinned
    return CugenReader


# ---------------------------------------------------------------------------
# Chromosome label helpers (verbatim port from step3_loco.py:31-54)
# ---------------------------------------------------------------------------
def extract_chr_num(filename):
    stem = Path(filename).stem
    rest = stem[3:]  # Remove "chr"
    parts = rest.split("_")
    return int(parts[0])


def extract_chr_label(filename):
    stem = Path(filename).stem
    rest = stem[3:]
    for suffix in ["_corrected"]:
        if suffix in rest:
            return rest.split(suffix)[0]
    return rest.split("_")[0]


def chr_sort_key(label):
    parts = label.split("_")
    chr_num = int(parts[0])
    suffix = int(parts[1]) if len(parts) > 1 else 0
    return (chr_num, suffix)


# ---------------------------------------------------------------------------
# Step-2 feather loader (verbatim port of load_snp_metadata)
# ---------------------------------------------------------------------------
def load_snp_metadata(step2_dir, verbose=True):
    pattern = os.path.join(step2_dir, "chr*_corrected.feather")
    feather_files = sorted(glob(pattern), key=lambda x: chr_sort_key(extract_chr_label(x)))

    if not feather_files:
        raise ValueError(f"No feather files found in {step2_dir}")

    if verbose:
        logger.info(f"Found {len(feather_files)} chromosome files")

    all_dfs = []
    for f in feather_files:
        chr_label = extract_chr_label(f)
        chr_num = extract_chr_num(f)
        df = pd.read_feather(f)
        df['chr_label'] = chr_label
        df['chr_num'] = chr_num
        df['chr'] = chr_num
        all_dfs.append(df)
        if verbose:
            logger.info(f"  chr{chr_label}: {len(df):,} SNPs")

    combined = pd.concat(all_dfs, ignore_index=True)
    if verbose:
        logger.info(f"Total: {len(combined):,} SNPs from {len(feather_files)} files")

    return combined


# ---------------------------------------------------------------------------
# Cugen genotype loader (verbatim port of load_genotypes_from_cugen)
# ---------------------------------------------------------------------------
def load_genotypes_from_cugen(snp_df, cugen_dir, device=0, n_extra_cols=0, verbose=True):
    """Stream selected genotypes directly to GPU. No CPU RAM for X."""
    Reader = _reader_cls()
    if verbose:
        logger.info("--- Loading genotypes from cugen files (direct to GPU) ---")
        logger.info(f"Cugen directory: {cugen_dir}  reader={Reader.__name__}")

    t0 = time.time()
    cp.cuda.Device(device).use()

    chr_groups = snp_df.groupby('chr_num')

    chr_info = {}
    all_dfs = []
    n_samples = None
    total_snps = 0

    for chr_num in sorted(chr_groups.groups.keys()):
        chr_snps = chr_groups.get_group(chr_num)

        cugen_path = os.path.join(cugen_dir, f"chr{chr_num}.cugen")
        if not os.path.exists(cugen_path):
            raise FileNotFoundError(f"Cugen file not found: {cugen_path}")

        reader = Reader(cugen_path, device=device)

        if n_samples is None:
            n_samples = reader.n_samples
        elif n_samples != reader.n_samples:
            raise ValueError(f"Sample count mismatch: {n_samples} vs {reader.n_samples}")

        cugen_gidx = reader.get_gidx()
        reader.close()

        gidx_to_local = {g: i for i, g in enumerate(cugen_gidx)}

        local_indices = []
        valid_gidx = []
        for gidx in chr_snps['gidx'].values:
            if gidx in gidx_to_local:
                local_indices.append(gidx_to_local[gidx])
                valid_gidx.append(gidx)

        if len(local_indices) == 0:
            continue

        sorted_pairs = sorted(zip(local_indices, valid_gidx))
        local_indices_sorted = [p[0] for p in sorted_pairs]
        valid_gidx_sorted = [p[1] for p in sorted_pairs]

        gidx_to_row = {row['gidx']: row for _, row in chr_snps.iterrows()}
        chr_df_rows = [gidx_to_row[g] for g in valid_gidx_sorted]
        chr_df = pd.DataFrame(chr_df_rows)
        all_dfs.append(chr_df)

        chr_info[chr_num] = (local_indices_sorted, len(local_indices_sorted))
        total_snps += len(local_indices_sorted)

    snp_df_ordered = pd.concat(all_dfs, ignore_index=True)

    total_cols = total_snps + n_extra_cols
    if verbose:
        logger.info(f"Pre-allocating GPU matrix: {n_samples:,} x {total_cols:,} "
                    f"({total_snps:,} SNPs + {n_extra_cols} extra)")
        logger.info(f"  GPU memory needed: {n_samples * total_cols * 4 / 1e9:.1f} GB")

    X_gpu = cp.zeros((n_samples, total_cols), dtype=cp.float32)

    col_offset = 0
    for chr_num in sorted(chr_info.keys()):
        local_indices_sorted, n_chr_snps = chr_info[chr_num]

        cugen_path = os.path.join(cugen_dir, f"chr{chr_num}.cugen")
        reader = Reader(cugen_path, device=device)

        if verbose:
            logger.info(f"  chr{chr_num}: reading {n_chr_snps} SNPs...")

        t_chr = time.time()
        # Session 13: coalesced-runs batched path (opt-in via USE_BATCHED_READ=1).
        if os.environ.get("USE_BATCHED_READ", "0") == "1" and hasattr(reader, "read_indices_to_gpu_batched"):
            X_chr = reader.read_indices_to_gpu_batched(local_indices_sorted)
        else:
            X_chr = reader.read_indices_to_gpu(local_indices_sorted)
        X_gpu[:, col_offset:col_offset + n_chr_snps] = X_chr

        col_offset += n_chr_snps
        del X_chr
        cp.get_default_memory_pool().free_all_blocks()

        if verbose:
            logger.info(f"  chr{chr_num}: {time.time() - t_chr:.1f}s")

        reader.close()

    load_time = time.time() - t0
    if verbose:
        logger.info(f"Loaded {total_snps:,} SNPs x {n_samples:,} samples in {load_time:.1f}s")

    return X_gpu, snp_df_ordered, total_snps, load_time


# ---------------------------------------------------------------------------
# Covariate loader (verbatim port of load_covariates)
# ---------------------------------------------------------------------------
def load_covariates(phe_file, fid_order, covar_list, null_markers, verbose=True):
    if verbose:
        logger.info(f"Loading covariates from {phe_file}")

    schema = {"FID": pl.Utf8}
    for c in covar_list:
        schema[c] = pl.Float64

    df = pl.read_csv(
        phe_file,
        separator="\t",
        columns=["FID", *covar_list],
        schema_overrides=schema,
        null_values=null_markers,
        infer_schema_length=10000,
        comment_prefix="#",
    )

    cov_map = {row["FID"]: [row.get(c) for c in covar_list] for row in df.iter_rows(named=True)}

    X_list = []
    missing = 0
    for fid in fid_order:
        vals = cov_map.get(fid, None)
        if vals is None or any(v is None for v in vals):
            X_list.append([np.nan] * len(covar_list))
            missing += 1
        else:
            X_list.append(vals)

    X_cov = np.asarray(X_list, dtype=np.float32)

    if missing > 0:
        logger.warning(f"{missing} samples missing covariates")

    if verbose:
        logger.info(f"Loaded {len(covar_list)} covariates: {covar_list}")
        logger.info(f"Covariate matrix shape: {X_cov.shape}")

    return X_cov


# ---------------------------------------------------------------------------
# Batched 2D @ 1D matmul helper.
# CuPy `X (f32 2D) @ v (f32 1D)` upcasts via tensordot to float64 → OOM
# at ~408K × 22K. We accumulate in column batches into a float32 buffer.
# Mirrors the inline loops in step3_loco.py:399-401 and :430-434.
# ---------------------------------------------------------------------------
def batched_linear_predictor(X_gpu, beta, indices_np=None, batch_size=4096):
    """Compute X[:, indices] @ beta in float32 batches.

    If ``indices_np`` is None, uses all columns of ``X_gpu`` and assumes
    ``beta.shape[0] == X_gpu.shape[1]``. Otherwise ``beta.shape[0] ==
    len(indices_np)``. Output is float32, length n_samples.
    """
    n_samples = X_gpu.shape[0]
    out = cp.zeros(n_samples, dtype=cp.float32)
    if indices_np is None:
        n_cols = X_gpu.shape[1]
        for bs in range(0, n_cols, batch_size):
            be = min(bs + batch_size, n_cols)
            out += X_gpu[:, bs:be] @ beta[bs:be]
    else:
        n_cols = len(indices_np)
        for bs in range(0, n_cols, batch_size):
            be = min(bs + batch_size, n_cols)
            out += X_gpu[:, indices_np[bs:be]] @ beta[bs:be]
    return out


# ---------------------------------------------------------------------------
# FISTA LASSO solver (verbatim port of gpu_fista_lasso)
# ---------------------------------------------------------------------------
def gpu_fista_lasso(X_gpu, y_gpu, alpha=1e-10, max_iter=1000, tol=1e-6, verbose=True):
    """GPU FISTA LASSO with L1 regularization."""
    n, p = X_gpu.shape

    if verbose:
        logger.info(f"Running GPU FISTA LASSO: {n:,} samples x {p:,} features")
        logger.info(f"  L1 alpha (LASSO): {alpha:.2e}")

    y_mean = float(cp.mean(y_gpu))
    y_c = y_gpu - y_mean

    if verbose:
        logger.info("Computing X^T X and X^T y...")
    t0 = time.time()
    XtX = cp.dot(X_gpu.T, X_gpu)
    Xty = cp.dot(X_gpu.T, y_c)
    precomp_time = time.time() - t0

    if verbose:
        logger.info(f"Precomputation completed in {precomp_time:.1f}s")

    # Lipschitz constant via power iteration
    if verbose:
        logger.info("Estimating Lipschitz constant...")
    v = cp.random.randn(p, dtype=cp.float32)
    v = v / cp.linalg.norm(v)
    for _ in range(30):
        v_new = XtX @ v
        v_new = v_new / cp.linalg.norm(v_new)
        v = v_new
    L = float(cp.dot(v, XtX @ v) / cp.dot(v, v)) / n
    L = max(L, 1e-10)

    step = 0.95 / L
    threshold = step * alpha

    if verbose:
        logger.info(f"  Lipschitz L={L:.2e}, step={step:.2e}, threshold={threshold:.2e}")

    theta = cp.zeros(p, dtype=cp.float32)
    theta_old = theta.copy()
    t = 1.0
    z = theta.copy()

    if verbose:
        logger.info("Running FISTA iterations...")
    t_iter = time.time()

    for k in range(max_iter):
        theta_old = theta.copy()
        grad = (XtX @ z - Xty) / n
        theta_new = z - step * grad
        theta = cp.sign(theta_new) * cp.maximum(cp.abs(theta_new) - threshold, 0)

        t_new = (1 + cp.sqrt(1 + 4 * t * t)) / 2
        z = theta + ((t - 1) / t_new) * (theta - theta_old)
        t = t_new

        if k % 50 == 0:
            diff = float(cp.linalg.norm(theta - theta_old))
            n_active = int(cp.sum(theta != 0))
            if verbose:
                logger.info(f"    iter {k}: ||delta||={diff:.2e}, active={n_active}")
            if diff < tol:
                if verbose:
                    logger.info(f"  Converged at iteration {k}")
                break

    iter_time = time.time() - t_iter
    n_active = int(cp.sum(theta != 0))

    # Store XtX and Xty for LOCO (before cleanup!)
    XtX_np = cp.asnumpy(XtX)
    Xty_np = cp.asnumpy(Xty)

    # Free GPU memory - XtX is now on CPU
    del XtX, Xty, z, v
    cp.get_default_memory_pool().free_all_blocks()

    # R² using batched matmul (avoid f32→f64 upcast OOM)
    theta = theta.astype(cp.float32)
    y_pred = batched_linear_predictor(X_gpu, theta) + y_mean
    ss_res = float(cp.sum((y_gpu - y_pred) ** 2))
    ss_tot = float(cp.sum((y_gpu - cp.mean(y_gpu)) ** 2))
    r2 = 1.0 - ss_res / ss_tot

    del y_pred
    cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        logger.info(f"  FISTA completed: {k+1} iters in {iter_time:.2f}s")
        logger.info(f"  Active features: {n_active}/{p}, R^2={r2:.6f}")

    return theta, {
        'iterations': k + 1,
        'iter_time': iter_time,
        'precomp_time': precomp_time,
        'n_active': n_active,
        'r2': r2,
        'intercept': y_mean,
        'XtX': XtX_np,
        'Xty': Xty_np,
    }


# ---------------------------------------------------------------------------
# Chromosome → feature-index mapping (verbatim port)
# ---------------------------------------------------------------------------
def build_chr_to_feature_indices(snp_df, n_snps, n_cov):
    chr_to_idx = {}
    for chr_num in range(1, 23):
        chr_mask = snp_df['chr_num'] == chr_num
        chr_indices = np.where(chr_mask.values)[0].tolist()
        chr_to_idx[chr_num] = chr_indices
    return chr_to_idx


# ---------------------------------------------------------------------------
# LOCO masked subtraction (verbatim port of loco_masked_predictions)
# ---------------------------------------------------------------------------
def loco_masked_predictions(X_gpu, y_gpu, theta_full, chr_to_idx, n_snps, n_cov,
                            verbose=True):
    """Compute LOCO predictions by subtracting each chromosome's contribution.

    Approximate (no re-fit), much faster than OLS. Used by ``loco_mode='lasso'``
    (a.k.a. ``'masked'`` in production CLI).
    """
    n_samples = X_gpu.shape[0]

    if verbose:
        logger.info("--- Computing LOCO predictions via masked subtraction ---")

    t0 = time.time()

    active_mask = cp.abs(theta_full) > 1e-10
    active_idx = cp.where(active_mask)[0]
    active_idx_np = cp.asnumpy(active_idx)
    n_active = len(active_idx)

    if verbose:
        logger.info(f"Active features from full model: {n_active}")

    theta_active = theta_full[active_idx]
    batch_size = 4096
    y_hat_full = batched_linear_predictor(X_gpu, theta_active,
                                          indices_np=active_idx_np,
                                          batch_size=batch_size)

    y_mean = float(cp.mean(y_gpu))
    y_hat_full += y_mean

    loco_predictions = np.zeros((n_samples, 22), dtype=np.float32)
    active_set = set(active_idx_np.tolist())

    if verbose:
        logger.info("Computing 22 LOCO predictions...")

    for chr_num in range(1, 23):
        t_chr = time.time()

        chr_orig_indices = chr_to_idx[chr_num]
        chr_active_orig = [idx for idx in chr_orig_indices if idx in active_set]
        n_chr_active = len(chr_active_orig)

        if n_chr_active == 0:
            loco_predictions[:, chr_num - 1] = cp.asnumpy(y_hat_full)
            if verbose:
                logger.info(f"  chr{chr_num}: 0 SNPs excluded, R²=N/A, "
                            f"{time.time() - t_chr:.3f}s")
            continue

        chr_theta = theta_full[chr_active_orig]
        chr_contribution = cp.zeros(n_samples, dtype=cp.float32)
        for bs in range(0, n_chr_active, batch_size):
            be = min(bs + batch_size, n_chr_active)
            batch_idx = chr_active_orig[bs:be]
            chr_contribution += X_gpu[:, batch_idx] @ chr_theta[bs:be]

        # f64 cast before subtraction avoids cancellation noise
        # (see article_reports/BUG_loco_f32_cancellation.md)
        y_pred = (y_hat_full.astype(cp.float64)
                  - chr_contribution.astype(cp.float64)).astype(cp.float32)
        loco_predictions[:, chr_num - 1] = cp.asnumpy(y_pred)

        if verbose:
            r2_loco = 1.0 - float(
                cp.sum((y_gpu - y_pred) ** 2) / cp.sum((y_gpu - y_mean) ** 2)
            )
            logger.info(f"  chr{chr_num}: {n_chr_active} SNPs excluded, "
                        f"R²={r2_loco:.4f}, {time.time() - t_chr:.3f}s")

        del chr_contribution, y_pred
        cp.get_default_memory_pool().free_all_blocks()

    total_time = time.time() - t0
    if verbose:
        logger.info(f"Total LOCO time: {total_time:.1f}s")

    return loco_predictions


# ---------------------------------------------------------------------------
# LOCO OLS refit (verbatim port of loco_ols_predictions — the production pipeline)
# ---------------------------------------------------------------------------
def loco_ols_predictions(X_gpu, y_gpu, theta_full, chr_to_idx, n_snps, n_cov,
                         XtX_full, Xty_full, ridge_lambda=1e-2, verbose=True):
    """Compute LOCO predictions via exact OLS refit on X'X subsetting.

    f64 cast + ``cp.linalg.solve`` + ridge λ=1e-2. Moore-Penrose pseudo-inverse
    over-drops signal columns — do NOT substitute. See CLAUDE.md dead-ends.
    """
    n_samples = X_gpu.shape[0]
    n_features = n_snps + n_cov

    if verbose:
        logger.info("--- Computing LOCO predictions via OLS ---")

    t0 = time.time()

    active_mask_full = cp.abs(theta_full) > 1e-10
    active_indices = cp.where(active_mask_full)[0]
    n_active = len(active_indices)

    if verbose:
        logger.info(f"Active features from full model: {n_active}")

    active_indices_np = cp.asnumpy(active_indices)

    XtX_active = XtX_full[np.ix_(active_indices_np, active_indices_np)]
    Xty_active = Xty_full[active_indices_np]

    cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        logger.info(f"Active features: {n_active}")
        logger.info(f"XtX_active shape: {XtX_active.shape}")

    orig_to_active = {int(orig): i for i, orig in enumerate(active_indices_np)}

    chr_to_active_idx = {}
    for chr_num in range(1, 23):
        chr_orig_indices = chr_to_idx[chr_num]
        chr_active = [orig_to_active[idx] for idx in chr_orig_indices
                      if idx in orig_to_active]
        chr_to_active_idx[chr_num] = chr_active
        if verbose and len(chr_active) > 0:
            logger.info(f"  chr{chr_num}: {len(chr_active)} active SNPs")

    y_mean = float(cp.mean(y_gpu))
    loco_predictions = np.zeros((n_samples, 22), dtype=np.float32)

    XtX_active_gpu = cp.asarray(XtX_active, dtype=cp.float32)
    Xty_active_gpu = cp.asarray(Xty_active, dtype=cp.float32)

    cov_orig_indices = list(range(n_snps, n_snps + n_cov))
    cov_active_indices = [orig_to_active[idx] for idx in cov_orig_indices
                          if idx in orig_to_active]

    if verbose:
        logger.info("Computing 22 LOCO predictions...")

    for chr_num in range(1, 23):
        t_chr = time.time()

        keep_mask = np.ones(n_active, dtype=bool)
        for idx in chr_to_active_idx[chr_num]:
            keep_mask[idx] = False

        keep_indices = np.where(keep_mask)[0]
        n_keep = len(keep_indices)

        if n_keep == 0:
            loco_predictions[:, chr_num - 1] = y_mean
            continue

        keep_idx_gpu = cp.asarray(keep_indices)
        XtX_loco = XtX_active_gpu[cp.ix_(keep_idx_gpu, keep_idx_gpu)]
        Xty_loco = Xty_active_gpu[keep_idx_gpu]

        n = n_samples
        # f64 cast + ridge=1e-2 — see step3_loco.py:568-572 for why this is
        # mandatory on near-singular 21K×21K XtX (full_wb chr15/19/21 R² <0).
        XtX_loco_f64 = XtX_loco.astype(cp.float64, copy=False)
        Xty_loco_f64 = Xty_loco.astype(cp.float64, copy=False)
        XtX_loco_f64 += ridge_lambda * n * cp.eye(n_keep, dtype=cp.float64)
        try:
            beta_loco = cp.linalg.solve(XtX_loco_f64, Xty_loco_f64).astype(cp.float32)
        except cp.linalg.LinAlgError:
            beta_loco = cp.linalg.lstsq(XtX_loco_f64, Xty_loco_f64,
                                        rcond=None)[0].astype(cp.float32)
        del XtX_loco_f64, Xty_loco_f64

        # Batched matmul to dodge fancy-index OOM (29 GB on full_wb imputed)
        final_indices = active_indices_np[keep_indices]
        y_pred = cp.full(n_samples, y_mean, dtype=cp.float32)

        batch_size = 2048  # tuned for H100 80 GB; lower if OOM
        for batch_start in range(0, len(final_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(final_indices))
            batch_idx = final_indices[batch_start:batch_end]
            beta_batch = beta_loco[batch_start:batch_end]
            y_pred += X_gpu[:, batch_idx] @ beta_batch

        loco_predictions[:, chr_num - 1] = cp.asnumpy(y_pred)

        if verbose:
            r2_loco = 1.0 - float(
                cp.sum((y_gpu - y_pred) ** 2) / cp.sum((y_gpu - y_mean) ** 2)
            )
            logger.info(f"  chr{chr_num}: {len(chr_to_active_idx[chr_num])} SNPs excluded, "
                        f"R²={r2_loco:.4f}, {time.time() - t_chr:.2f}s")

        del XtX_loco, Xty_loco, beta_loco, keep_idx_gpu, y_pred
        cp.get_default_memory_pool().free_all_blocks()

    total_time = time.time() - t0
    if verbose:
        logger.info(f"Total LOCO time: {total_time:.1f}s")

    return loco_predictions


# ---------------------------------------------------------------------------
# Default covariate list (the production pipeline)
# ---------------------------------------------------------------------------
_DEFAULT_COVARS = ['age', 'sex'] + [f'PC{i}' for i in range(1, 11)]
_DEFAULT_MASTER_PHE = None  # caller must supply a path
_DEFAULT_NULLS = ["NA", "NaN", "nan", ""]


# ---------------------------------------------------------------------------
# Public API: fit_joint_lasso
# ---------------------------------------------------------------------------
def fit_joint_lasso(
    candidates,
    cohort_npz: Union[str, Path],
    cugen_dir: Union[str, Path],
    *,
    covariates: Optional[Sequence[str]] = None,
    alpha: float = 6e-3,
    ridge: float = 1e-2,
    loco_mode: str = "ols",
    output_dir: Optional[Union[str, Path]] = None,
    # extras (default to production behaviour, not in public signature)
    phe_file: Optional[Union[str, Path]] = None,
    null_markers: Optional[Sequence[str]] = None,
    device: int = 0,
    max_iter: int = 1000,
    verbose: bool = True,
) -> dict:
    """Fit joint LASSO across all chromosomes and emit LOCO residuals.

    Parameters
    ----------
    candidates : DataFrame, str, or Path
        Output of screen_chromosome aggregated genome-wide. Accepted forms:
          * pandas.DataFrame with columns including ['gidx', 'chr_num']
          * a path to a step-2 output directory containing chr*_corrected.feather
            (production layout — same path passed to ``step3_loco.py --step2-dir``)
    cohort_npz : path
        NPZ written by prepare_cohort (residualised cohort).
    cugen_dir : path
        Directory holding chr{1..22}.cugen.
    covariates : sequence of str
        Covariate column names in phe_file. Default: age, sex, PC1..10.
    alpha : float
        Step-3 FISTA L1 penalty (default 6e-3 = the production pipeline).
    ridge : float
        Ridge regularisation for LOCO OLS refit (default 1e-2 = Fix-I).
    loco_mode : {'ols', 'lasso', 'masked'}
        'ols' (the production pipeline, default) or 'lasso'/'masked' (legacy v5).
    output_dir : path, optional
        If set, write ``full_model.feather`` + ``loco_predictions.npz`` here.
    phe_file : path, optional
        Master phenotype TSV (default = production OAK master.phe).
    null_markers : sequence of str, optional
        Strings to treat as NA in phe_file.
    device : int
        CUDA device (default 0).
    max_iter : int
        FISTA max iterations (default 1000).
    verbose : bool

    Returns
    -------
    dict with keys:
        active_set        : DataFrame — full SNP model (step3_coef_norm, step3_coef,
                            snp_mean, snp_std, step3_active, chr_num, gidx, ...)
        loco_predictions  : np.ndarray (n_samples, 22) — per-chromosome predicted ŷ
        cv_metrics        : {'r2', 'n_active', 'iterations', 'iter_time',
                             'precomp_time', 'intercept', 'load_time',
                             'fista_time', 'loco_time'}
        fid_order         : np.ndarray of FIDs aligned to loco_predictions rows
        y_original        : np.ndarray of phenotype values aligned to fid_order
    """
    # Normalize loco_mode: legacy CLI says 'masked'; public API also accepts 'lasso'.
    mode = loco_mode.lower()
    if mode == "lasso":
        mode = "masked"
    if mode not in ("ols", "masked"):
        raise ValueError(f"loco_mode must be 'ols' or 'lasso'/'masked', got {loco_mode!r}")

    covariates = list(covariates) if covariates is not None else list(_DEFAULT_COVARS)
    phe_file = str(phe_file) if phe_file is not None else _DEFAULT_MASTER_PHE
    null_markers = list(null_markers) if null_markers is not None else list(_DEFAULT_NULLS)

    cp.cuda.Device(device).use()

    if verbose:
        logger.info("=" * 60)
        logger.info("STEP 3 LOCO: Leave-One-Chromosome-Out Training")
        mode_desc = "exact OLS via X'X subsetting" if mode == "ols" else "masked subtraction"
        logger.info(f"  Mode: {mode_desc}")
        logger.info(f"  alpha={alpha:.2e}  ridge={ridge:.2e}")
        logger.info("=" * 60)
        free, total = cp.cuda.runtime.memGetInfo()
        try:
            props = cp.cuda.runtime.getDeviceProperties(device)
            gpu_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
            logger.info(f"  GPU: {gpu_name}  VRAM: {free/1e9:.1f}/{total/1e9:.1f} GB free")
        except Exception:  # noqa: BLE001
            logger.info(f"  VRAM: {free/1e9:.1f}/{total/1e9:.1f} GB free")

    if output_dir is not None:
        os.makedirs(str(output_dir), exist_ok=True)

    # ---- Load cohort ----
    cohort = np.load(str(cohort_npz), allow_pickle=True)
    if 'y_original' in cohort.files:
        y = cohort['y_original'].astype(np.float32)
        if verbose:
            logger.info("Using RAW phenotype (y_original)")
    else:
        y = cohort['y_train'].astype(np.float32)
        if verbose:
            logger.info("Using y_train from cohort")

    fid_order = cohort['fid_order'].astype(str)
    n_samples = len(y)
    if verbose:
        logger.info(f"Loaded cohort: {n_samples:,} samples")
        logger.info(f"Phenotype stats: mean={y.mean():.2f}, std={y.std():.2f}")

    # ---- Covariates ----
    if verbose:
        logger.info("--- Loading covariates ---")
    X_cov = load_covariates(phe_file, fid_order, covariates, null_markers, verbose=verbose)
    n_cov = X_cov.shape[1]

    # ---- SNP metadata: accept either a DataFrame or a step2-dir path ----
    if isinstance(candidates, pd.DataFrame):
        snp_df = candidates.copy()
        # Ensure required columns are present (chr_num at minimum)
        if 'chr_num' not in snp_df.columns:
            if 'chrom' in snp_df.columns:
                snp_df['chr_num'] = snp_df['chrom'].astype(int)
            elif 'chr' in snp_df.columns:
                snp_df['chr_num'] = snp_df['chr'].astype(int)
            else:
                raise KeyError("candidates DataFrame needs a 'chr_num', 'chrom', or 'chr' column")
        if 'chr' not in snp_df.columns:
            snp_df['chr'] = snp_df['chr_num']
        if 'chr_label' not in snp_df.columns:
            snp_df['chr_label'] = snp_df['chr_num'].astype(str)
        if 'gidx' not in snp_df.columns:
            raise KeyError("candidates DataFrame needs a 'gidx' column to index cugen variants")
    else:
        if verbose:
            logger.info("--- Loading SNP metadata from step2 dir ---")
        snp_df = load_snp_metadata(str(candidates), verbose=verbose)

    # ---- Load genotypes ----
    X_gpu, snp_df, n_snps, load_time = load_genotypes_from_cugen(
        snp_df, str(cugen_dir), device=device, n_extra_cols=n_cov, verbose=verbose
    )
    n_features = n_snps + n_cov

    # ---- Normalize SNPs on GPU in-place ----
    if verbose:
        logger.info("--- Normalizing SNPs (on GPU, in-place) ---")
    X_snps_view = X_gpu[:, :n_snps]
    snp_means = X_snps_view.mean(axis=0)
    snp_stds = X_snps_view.std(axis=0)
    snp_stds = cp.where(snp_stds < 1e-6, 1.0, snp_stds)
    X_snps_view -= snp_means
    X_snps_view /= snp_stds

    snp_means_np = cp.asnumpy(snp_means)
    snp_stds_np = cp.asnumpy(snp_stds)
    del snp_means, snp_stds
    cp.get_default_memory_pool().free_all_blocks()

    # ---- Normalize covariates and load into trailing columns ----
    if n_cov > 0:
        if verbose:
            logger.info("Normalizing and adding covariates...")
        cov_means = np.nanmean(X_cov, axis=0)
        cov_stds = np.nanstd(X_cov, axis=0)
        cov_stds[cov_stds == 0] = 1.0
        X_cov_norm = (X_cov - cov_means) / cov_stds
        X_cov_norm = np.nan_to_num(X_cov_norm, nan=0.0)
        X_gpu[:, n_snps:n_features] = cp.asarray(X_cov_norm, dtype=cp.float32)
        del X_cov, X_cov_norm

    y_gpu = cp.asarray(y, dtype=cp.float32)

    if verbose:
        logger.info(f"Combined: {n_features:,} features ({n_snps:,} SNPs + {n_cov} covariates)")
        logger.info(f"Matrix size: {X_gpu.nbytes / 1e9:.1f} GB (on GPU)")

    cp.get_default_memory_pool().free_all_blocks()

    chr_to_idx = build_chr_to_feature_indices(snp_df, n_snps, n_cov)

    # ---- FISTA ----
    if verbose:
        logger.info("--- Running FISTA on full data (all chromosomes) ---")
    t_fista = time.time()
    theta_full, info = gpu_fista_lasso(X_gpu, y_gpu, alpha=alpha, max_iter=max_iter, verbose=verbose)
    fista_time = time.time() - t_fista
    cp.get_default_memory_pool().free_all_blocks()

    # ---- Full-model feather ----
    coef_np = cp.asnumpy(theta_full)
    snp_coefs_norm = coef_np[:n_snps]

    snp_df['step3_coef_norm'] = snp_coefs_norm
    snp_df['step3_coef'] = snp_coefs_norm / snp_stds_np
    snp_df['snp_mean'] = snp_means_np
    snp_df['snp_std'] = snp_stds_np
    snp_df['step3_active'] = np.abs(snp_coefs_norm) > 1e-10

    if output_dir is not None:
        full_model_path = os.path.join(str(output_dir), "full_model.feather")
        snp_df.to_feather(full_model_path)
        if verbose:
            logger.info(f"Saved full model to {full_model_path}")

    # ---- LOCO ----
    t_loco = time.time()
    if mode == "masked":
        loco_predictions = loco_masked_predictions(
            X_gpu, y_gpu, theta_full, chr_to_idx, n_snps, n_cov,
            verbose=verbose,
        )
    else:
        loco_predictions = loco_ols_predictions(
            X_gpu, y_gpu, theta_full, chr_to_idx, n_snps, n_cov,
            info['XtX'], info['Xty'], ridge_lambda=ridge, verbose=verbose,
        )
    loco_time = time.time() - t_loco

    if output_dir is not None:
        loco_path = os.path.join(str(output_dir), "loco_predictions.npz")
        np.savez_compressed(
            loco_path,
            predictions=loco_predictions,
            fid_order=fid_order,
            y_original=y,
        )
        if verbose:
            logger.info(f"Saved LOCO predictions to {loco_path}  shape={loco_predictions.shape}")

    cv_metrics = {
        'r2': info['r2'],
        'n_active': info['n_active'],
        'iterations': info['iterations'],
        'iter_time': info['iter_time'],
        'precomp_time': info['precomp_time'],
        'intercept': info['intercept'],
        'load_time': load_time,
        'fista_time': fista_time,
        'loco_time': loco_time,
    }

    if verbose:
        logger.info("=" * 60)
        logger.info("RESULTS SUMMARY")
        logger.info(f"  L1 alpha: {alpha:.2e}  ridge: {ridge:.2e}  mode: {mode}")
        logger.info(f"  R² = {info['r2']:.6f}")
        logger.info(f"  Active SNPs: {info['n_active']:,}/{n_features:,}")
        logger.info(f"  Cugen load: {load_time:.1f}s   FISTA: {fista_time:.1f}s   LOCO: {loco_time:.1f}s")
        logger.info("=" * 60)

    # Release main genotype matrix before returning
    del X_gpu, y_gpu, theta_full
    cp.get_default_memory_pool().free_all_blocks()

    return {
        'active_set': snp_df,
        'loco_predictions': loco_predictions,
        'cv_metrics': cv_metrics,
        'fid_order': fid_order,
        'y_original': y,
    }
