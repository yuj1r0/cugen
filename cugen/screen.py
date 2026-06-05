"""cugen.screen — Step 1+2: per-chromosome block-LASSO SNP screening.

Verbatim port of ``step1_to_step2_cugen_windowed.py`` (production windowed
Step 1+2). The streaming loop, TopK buffer, LOO build, and FISTA solver
are byte-exact copies of the production logic — only the SLURM/argparse
boilerplate has been dropped in favour of the package's function API.

Pipeline:
  * Stream chromosome variants in IO blocks; compute univariate β/se on GPU.
  * Maintain N TopK buffers (one per window) by SE-weighted strength
    ``slope² * maf * (1-maf) * n * sxx / syy``  (= r² / SE²).
  * Combine buffers, build LOO predictions, run positive-FISTA LASSO.
  * Return (and optionally write to feather) the selected variants.

Env-vars honoured:
  * ``USE_PINNED_READER=1`` swaps :class:`CugenReader` → :class:`CugenReaderPinned`
    for ~1.6-1.85× sequential-read throughput.
"""

import gc
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional, Union

import cupy as cp
import numpy as np
import pandas as pd

# Reader selection: mirrors what production sitecustomize.py does at import time.
if os.environ.get("USE_PINNED_READER", "0") == "1":
    from .io import CugenReaderPinned as CugenReader  # noqa: F401
else:
    from .io import CugenReader  # noqa: F401


log = logging.getLogger("cugen.screen")


# =========================================================================
# FISTA positive LASSO — verbatim from production step1_to_step2_cugen_windowed.py
# =========================================================================
def gpu_positive_lasso_fista(X_gpu, y_gpu, alpha, max_iter=1000, tol=1e-6, verbose=True):
    """GPU-native positive LASSO using FISTA (Fast Iterative Shrinkage-Thresholding)."""
    n, p = X_gpu.shape

    y_mean = float(cp.mean(y_gpu))
    y_c = y_gpu - y_mean

    t0 = time.time()
    XtX = cp.dot(X_gpu.T, X_gpu)
    Xty = cp.dot(X_gpu.T, y_c)
    precomp_time = time.time() - t0

    if verbose:
        log.info("  Precomputed X^T X and X^T y in %.2fs", precomp_time)

    # Lipschitz constant via power iteration
    v = cp.random.randn(p, dtype=cp.float32)
    v = v / cp.linalg.norm(v)
    for _ in range(20):
        v_new = XtX @ v
        v_new = v_new / cp.linalg.norm(v_new)
        v = v_new
    L = float(cp.dot(v, XtX @ v) / cp.dot(v, v))
    L = L / n

    step = 0.95 / L
    threshold = step * alpha

    if verbose:
        log.info("  Lipschitz L=%.2e, step=%.2e, threshold=%.2e", L, step, threshold)

    theta = cp.zeros(p, dtype=cp.float32)
    theta_old = theta.copy()
    t = 1.0
    z = theta.copy()

    t_iter = time.time()
    for k in range(max_iter):
        theta_old = theta.copy()
        grad = (XtX @ z - Xty) / n
        theta_new = z - step * grad
        theta = cp.maximum(theta_new - threshold, 0)

        t_new = (1 + cp.sqrt(1 + 4 * t * t)) / 2
        z = theta + ((t - 1) / t_new) * (theta - theta_old)
        t = t_new

        if k % 10 == 0:
            diff = float(cp.linalg.norm(theta - theta_old))
            n_active = int(cp.sum(theta > 0))
            if verbose and k % 50 == 0:
                log.info("    iter %d: ||dtheta||=%.2e, active=%d", k, diff, n_active)
            if diff < tol:
                if verbose:
                    log.info("  Converged at iteration %d", k)
                break

    iter_time = time.time() - t_iter

    X_mean = cp.mean(X_gpu, axis=0)
    intercept = float(y_mean - cp.dot(X_mean, theta))

    y_pred = X_gpu @ theta + intercept
    ss_res = float(cp.sum((y_gpu - y_pred) ** 2))
    ss_tot = float(cp.sum((y_gpu - y_mean) ** 2))
    r2 = 1 - ss_res / ss_tot

    n_active = int(cp.sum(theta > 0))

    if verbose:
        log.info("  FISTA completed: %d iters in %.2fs", k + 1, iter_time)
        log.info("  Active features: %d/%d, R2=%.6f", n_active, p, r2)

    return theta, intercept, {
        "iterations": k + 1,
        "precomp_time": precomp_time,
        "iter_time": iter_time,
        "r2": r2,
        "n_active": n_active,
    }


class TopKBuffer:
    """Maintains top-K SNPs on GPU with their genotypes and statistics."""

    def __init__(self, k, n_samples, device=0):
        self.k = k
        self.n_samples = n_samples
        self.device = device

        cp.cuda.Device(device).use()

        self.genotypes = cp.zeros((n_samples, k), dtype=cp.float32)
        self.gidx = np.full(k, -1, dtype=np.int64)
        self.mu_x = np.zeros(k, dtype=np.float32)
        self.den = np.zeros(k, dtype=np.float32)
        self.alpha = np.zeros(k, dtype=np.float64)
        self.beta = np.zeros(k, dtype=np.float64)
        self.strength = np.zeros(k, dtype=np.float64)

        self.count = 0
        self.min_idx = 0
        self.min_strength = 0.0

    def _find_min(self):
        if self.count == 0:
            return 0, 0.0
        valid_strengths = self.strength[:self.count]
        self.min_idx = int(np.argmin(valid_strengths))
        self.min_strength = valid_strengths[self.min_idx]
        return self.min_idx, self.min_strength

    def update(self, gidx_batch, genotypes_gpu, mu_batch, den_batch,
               alpha_batch, beta_batch, strength_batch):
        batch_size = len(gidx_batch)

        for i in range(batch_size):
            strength_i = strength_batch[i]

            if self.count < self.k:
                idx = self.count
                self.gidx[idx] = gidx_batch[i]
                self.mu_x[idx] = mu_batch[i]
                self.den[idx] = den_batch[i]
                self.alpha[idx] = alpha_batch[i]
                self.beta[idx] = beta_batch[i]
                self.strength[idx] = strength_i
                self.genotypes[:, idx] = genotypes_gpu[:, i]
                self.count += 1

                if self.count == self.k:
                    self._find_min()
            else:
                if strength_i > self.min_strength:
                    idx = self.min_idx
                    self.gidx[idx] = gidx_batch[i]
                    self.mu_x[idx] = mu_batch[i]
                    self.den[idx] = den_batch[i]
                    self.alpha[idx] = alpha_batch[i]
                    self.beta[idx] = beta_batch[i]
                    self.strength[idx] = strength_i
                    self.genotypes[:, idx] = genotypes_gpu[:, i]
                    self._find_min()

    def get_sorted(self):
        if self.count == 0:
            return None, None

        order = np.argsort(-self.strength[:self.count])
        order_gpu = cp.asarray(order)
        genotypes_sorted = self.genotypes[:, order_gpu]

        stats_df = pd.DataFrame({
            "gidx": self.gidx[:self.count][order],
            "mu_x": self.mu_x[:self.count][order],
            "den": self.den[:self.count][order],
            "alpha": self.alpha[:self.count][order],
            "beta": self.beta[:self.count][order],
            "strength": self.strength[:self.count][order],
        })

        return genotypes_sorted, stats_df


def build_loo_and_run_lasso(genotypes_gpu, stats_df, y_gpu, alpha_lasso=1e-05, device=0):
    """Build LOO matrix and run GPU LASSO."""
    n, K = genotypes_gpu.shape
    cp.cuda.Device(device).use()

    log.info("[CUGEN] Building LOO from GPU genotypes: %d samples x %d SNPs", n, K)

    inv_n = 1.0 / n

    mu_all = cp.asarray(stats_df["mu_x"].to_numpy(np.float32))
    den_all = cp.maximum(cp.asarray(stats_df["den"].to_numpy(np.float32)), 1e-20)
    alpha_all = cp.asarray(stats_df["alpha"].to_numpy(np.float32))
    beta_all = cp.asarray(stats_df["beta"].to_numpy(np.float32))

    t0 = time.time()
    # Memory-optimized LOO computation
    # 1. Compute centered genotypes, then delete local reference.
    # NOTE: del only removes local ref; caller's ref keeps array alive during function.
    # The array will be freed when caller's ref goes out of scope (after function returns).
    xc = genotypes_gpu - mu_all[None, :]
    del genotypes_gpu
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    # 2. Compute leverage (h) values
    h = inv_n + (xc * xc) / den_all[None, :]

    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    # 3. fitted = alpha + beta*X = (alpha + beta*mu) + beta*xc
    fitted = (alpha_all + beta_all * mu_all)[None, :] + beta_all[None, :] * xc
    del xc
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    # 4. LOO predictions
    LOO_gpu = y_gpu[:, None] - ((y_gpu[:, None] - fitted) / (1.0 - h))
    del h, fitted
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    loo_time = time.time() - t0
    log.info("  LOO built in %.1fs", loo_time)

    log.info("[CUGEN] Running GPU FISTA Positive LASSO (alpha=%g)...", alpha_lasso)
    t0 = time.time()

    theta_gpu, theta_0, lasso_info = gpu_positive_lasso_fista(
        LOO_gpu, y_gpu, alpha=alpha_lasso, max_iter=1000, tol=1e-6, verbose=True
    )

    lasso_time = time.time() - t0
    log.info("  GPU LASSO total time: %.2fs", lasso_time)

    theta_cpu = cp.asnumpy(theta_gpu).astype(np.float32)

    univariate_intercepts = stats_df["alpha"].values
    univariate_slopes = stats_df["beta"].values
    final_intercept = theta_0 + np.dot(theta_cpu, univariate_intercepts)
    final_slopes = theta_cpu * univariate_slopes

    r2 = lasso_info["r2"]
    n_selected = np.sum(theta_cpu > 0)
    n_pos = np.sum(final_slopes > 0)
    n_neg = np.sum(final_slopes < 0)

    log.info("  Results: %d/%d selected, R2 = %.6f", n_selected, K, r2)
    log.info("  Final coeffs: %d+ %d-", n_pos, n_neg)

    return {
        "theta": theta_cpu,
        "theta_0": theta_0,
        "final_intercept": final_intercept,
        "final_slopes": final_slopes,
        "r2": r2,
        "n_selected": n_selected,
    }


def choose_batch_size(n_samples, frac_free=0.25, max_batch=8000):
    free, _ = cp.cuda.runtime.memGetInfo()
    bytes_per_variant = n_samples * 16
    auto = max(256, int(frac_free * free // bytes_per_variant))
    return int(max(256, min(max_batch, auto)))


# =========================================================================
# Public API
# =========================================================================
STRENGTH_FUNCTIONS = {}


def register_strength_fn(name):
    """Decorator: register a named strength function for ``screen_chromosome``.

    A strength function takes a dict of per-variant univariate statistics
    and returns a 1D numpy array (same length) of "strength" scores. The
    TopK buffer keeps variants with the highest strength.

    ``stats`` dict keys (all 1D numpy arrays, same length per call):
      ``slope``       — univariate OLS slope (β̂_j)
      ``intercept``   — univariate OLS intercept
      ``maf``         — minor-allele frequency, header value
      ``sxx``         — Σ(x - x̄)² for the variant, header value
      ``syy``         — Σ(y - ȳ)² for the phenotype (scalar broadcast)
      ``n``           — sample count (scalar broadcast)
      ``mu_x``        — per-variant mean genotype
    """
    def deco(fn):
        STRENGTH_FUNCTIONS[name] = fn
        return fn
    return deco


@register_strength_fn("r2_se")
def _strength_r2_over_se2(stats):
    """Default. ``β² · maf · (1-maf) · n · sxx / syy`` (= r² / SE², the
    SE-weighted variance-explained measure used in the production pipeline)."""
    return ((stats["slope"] ** 2) * stats["maf"] * (1.0 - stats["maf"])
            * (stats["n"] * stats["sxx"] / stats["syy"]))


@register_strength_fn("r2")
def _strength_r2(stats):
    """In-sample univariate r²: ``β² · sxx / syy``."""
    return (stats["slope"] ** 2) * stats["sxx"] / stats["syy"]


@register_strength_fn("beta_abs")
def _strength_beta_abs(stats):
    """``|β̂|`` — raw effect-size magnitude."""
    import numpy as np
    return np.abs(stats["slope"])


@register_strength_fn("z_abs")
def _strength_z_abs(stats):
    """``|z|`` under H0 σ̂²_y/n variance: ``|β̂| · sqrt(n · sxx / syy)``."""
    import numpy as np
    return np.abs(stats["slope"]) * np.sqrt(
        stats["n"] * stats["sxx"] / stats["syy"]
    )


@register_strength_fn("chi2")
def _strength_chi2(stats):
    """``z²`` (= n · sxx · β² / syy). Equivalent ranking to ``z_abs``,
    different scale."""
    return ((stats["slope"] ** 2) * stats["n"] * stats["sxx"] / stats["syy"])


def screen_chromosome(
    chrom,
    cohort_npz,
    cugen_path,
    *,
    block_size: int = 8192,
    block_pct: float = 20.0,
    alpha: float = 6e-4,
    maf_min: float = 1e-2,
    strength_fn=None,
    output: Optional[Union[str, Path]] = None,
    output_suffix: str = "",
    # advanced (pass-through to production logic — keep prod defaults)
    device: int = 0,
    io_block: int = 8192,
    max_batch: int = 8000,
    lam_quantile: float = 5.0,
    variant_start: int = 0,
    variant_end: Optional[int] = None,
    show_block_times: bool = False,
    save_npy: bool = False,
    npy_dir: Optional[Union[str, Path]] = None,
    return_candidates: bool = False,
) -> pd.DataFrame:
    """Run Step 1+2 windowed LASSO screening on one chromosome.

    Parameters
    ----------
    chrom : int or str
        Chromosome label (used only for output filename / logging).
    cohort_npz : str, Path, or dict-like
        Path to residualised cohort NPZ (output of :func:`prepare_cohort`),
        or an already-loaded dict / NpzFile providing ``y_train``.
    cugen_path : str or Path
        Path to ``chr{chrom}.cugen``.
    block_size : int
        Block size for block-based selection (variants per window). Default 8192.
    block_pct : float
        Percentage of each block to retain in the TopK buffer. Default 20.0.
    alpha : float
        FISTA LASSO penalty. Default 6e-4 (production continuous).
    maf_min : float
        MAF floor. Default 1e-2 (production).
    strength_fn : callable or str, optional
        Ranking function for the per-window TopK buffer. Accepts either a
        string from :data:`STRENGTH_FUNCTIONS` (``'r2_se'`` [default,
        production], ``'r2'``, ``'beta_abs'``, ``'z_abs'``, ``'chi2'``) or a
        callable mapping a stats dict → 1-D numpy array of strengths. See
        :func:`register_strength_fn` for the dict-keys contract.
    output : str or Path, optional
        If given, the result DataFrame is written as feather to this path.
        If ``output_suffix`` is non-empty, it is appended before the extension.
    output_suffix : str
        Suffix appended to ``output`` filename stem (e.g. "_1" for split-half).
    device : int
        CUDA device id (default 0).
    io_block : int
        Per-read I/O block size in variants (default 8192).
    max_batch : int
        Max GPU compute batch (default 8000).
    lam_quantile : float
        Percentile of sxx used as regularisation lambda (default 5.0).
    variant_start, variant_end : int
        Optional variant range (for splitting large chr).
    show_block_times : bool
        Log per-block timings if True.
    save_npy : bool
        If True, write selected genotypes as ``.npy`` to ``npy_dir`` (or
        the output directory). Default False (the cugen-aware Step 3 reads
        genotypes directly from the cugen).
    npy_dir : str or Path, optional
        Output directory for the ``.npy`` file. Defaults to ``output``'s parent.

    Returns
    -------
    pandas.DataFrame
        Selected variants with columns
        ``[gidx, mu_x, den, alpha, beta, strength, window,
           lasso_weight, final_coef_corrected]``.
    """
    log.info("=" * 70)
    log.info("FUSED STEP 1 + STEP 2: CugenReader Edition - WINDOWED")
    log.info("=" * 70)
    chr_label = f"{chrom}{output_suffix}" if output_suffix else str(chrom)
    log.info("Chromosome: %s", chr_label)
    if variant_end is not None or variant_start > 0:
        log.info("Variant range: %d - %s", variant_start, variant_end)
    log.info(
        "Selection mode: BLOCK-BASED (%d variants/block, top %.2f%% per block)",
        block_size, block_pct,
    )
    log.info("LASSO alpha: %g", alpha)

    cp.cuda.Device(device).use()
    gpu_name = cp.cuda.runtime.getDeviceProperties(device)["name"].decode()
    free, total = cp.cuda.runtime.memGetInfo()
    log.info("GPU: %s", gpu_name)
    log.info("VRAM: %.1f/%.1f GB", free / 1e9, total / 1e9)

    total_start = time.time()

    # Load cohort — accept either a path or a dict-like object.
    if isinstance(cohort_npz, (str, Path)):
        cohort_data = np.load(str(cohort_npz), allow_pickle=True)
    else:
        cohort_data = cohort_npz
    y_train = np.asarray(cohort_data["y_train"]).astype(np.float32)
    n = len(y_train)

    # Resolve strength function (string name → callable, or pass through).
    if strength_fn is None or isinstance(strength_fn, str):
        key = strength_fn or "r2_se"
        if key not in STRENGTH_FUNCTIONS:
            raise ValueError(
                f"Unknown strength_fn={key!r}. Available: "
                f"{sorted(STRENGTH_FUNCTIONS)}"
            )
        _strength_callable = STRENGTH_FUNCTIONS[key]
        log.info("  Strength function: %s", key)
    elif callable(strength_fn):
        _strength_callable = strength_fn
        log.info("  Strength function: <user callable> (%s)",
                 getattr(strength_fn, "__name__", "anonymous"))
    else:
        raise TypeError(
            "strength_fn must be a registered name (str), a callable, or None"
        )

    # Open cugen file
    log.info("Opening %s...", cugen_path)
    reader = CugenReader(str(cugen_path), device=device)
    info = reader.info()
    log.info("  Samples: %d", info["n_samples"])
    log.info("  Variants: %d", info["n_variants"])
    log.info("  Encoding: %s", info["encoding"])
    log.info("  Size: %.2f GB", info["file_size_gb"])

    # Get precomputed stats
    t_stats = time.time()
    mu_x, sxx, maf = reader.get_stats()
    original_gidx = reader.get_gidx()
    log.info("  Stats loaded in %.1fms", (time.time() - t_stats) * 1000)

    # Apply MAF + sxx filtering
    maf_mask = maf >= maf_min
    sxx_mask = sxx > 0
    keep_mask = maf_mask & sxx_mask
    keep_indices = np.where(keep_mask)[0]

    # Apply variant range filter if specified
    if variant_end is not None or variant_start > 0:
        var_start = variant_start
        var_end = variant_end if variant_end is not None else info["n_variants"]
        range_mask = (keep_indices >= var_start) & (keep_indices < var_end)
        keep_indices = keep_indices[range_mask]
        log.info("  Variant range filter: %d - %d", var_start, var_end)

    kept = len(keep_indices)
    log.info("  After filtering: %d SNPs (MAF >= %g, sxx > 0)", kept, maf_min)

    # Lambda for regularization
    lam = float(np.percentile(sxx[keep_mask], lam_quantile)) if lam_quantile > 0 else 0.0
    log.info("  Lambda (regularization): %.2f", lam)

    # Choose batch size (informational — kernel size; production uses io_block to drive)
    batch_size = choose_batch_size(n, 0.25, max_batch)
    log.info("  GPU batch size: %d", batch_size)

    # GPU arrays
    y_gpu = cp.asarray(y_train)
    y_bar = float(cp.mean(y_gpu))
    y_c = y_gpu - y_bar
    syy = float(cp.sum(y_c ** 2))
    inv_n = 1.0 / n
    log.info("  syy (total variance in y): %.2f", syy)

    # Block-based mode
    window_size = block_size
    n_windows = (kept + window_size - 1) // window_size  # ceiling division
    k_per_window = max(50, int(window_size * block_pct / 100))
    log.info("  Block-based selection: %d blocks of %d variants", n_windows, window_size)
    log.info("  Keeping top %.2f%% per block = %d SNPs/block", block_pct, k_per_window)

    if n_windows == 0:
        log.warning("No variants survive filtering — returning empty DataFrame")
        reader.close()
        return pd.DataFrame(columns=[
            "gidx", "mu_x", "den", "alpha", "beta", "strength",
            "window", "lasso_weight", "final_coef_corrected",
        ])

    topk_buffers = [TopKBuffer(k_per_window, n, device=device) for _ in range(n_windows)]

    total_buffer_mem = n_windows * k_per_window * n * 4 / 1e9
    log.info(
        "  Top-K buffers: %d windows x %d SNPs/window = %d max total",
        n_windows, k_per_window, n_windows * k_per_window,
    )
    log.info("  Buffer memory: %.2f GB on GPU", total_buffer_mem)

    # ============================================================
    # MAIN LOOP — stream filtered SNPs through CugenReader
    # ============================================================
    log.info("[PASS 1+2] Streaming %d SNPs using CugenReader...", kept)
    t_stream = time.time()

    base = 0
    total_io_time = 0.0
    total_compute_time = 0.0

    X_gpu = None
    mu_gpu = None
    den_gpu = None
    x_c = None
    slope = None
    intercept = None

    while base < kept:
        end = min(base + io_block, kept)
        block_indices = keep_indices[base:end]
        take = len(block_indices)

        # Determine contiguous ranges for efficient reading
        start_idx = int(block_indices[0])
        end_idx = int(block_indices[-1]) + 1

        # Read genotypes from cugen (GPU-accelerated)
        t0 = time.time()
        X_gpu_full = reader.read_to_gpu(start_idx, end_idx)
        io_s = time.time() - t0
        total_io_time += io_s

        local_indices = block_indices - start_idx
        X_gpu = X_gpu_full[:, local_indices]
        del X_gpu_full

        # Block stats
        block_gidx = original_gidx[block_indices]
        block_mu = mu_x[block_indices]
        block_sxx = sxx[block_indices]
        block_maf = maf[block_indices]
        block_den = block_sxx + lam

        # Univariate stats on GPU
        t1 = time.time()

        mu_gpu = cp.asarray(block_mu)
        den_gpu = cp.asarray(block_den)

        x_c = X_gpu - mu_gpu[None, :]
        slope = cp.sum(x_c * y_c[:, None], axis=0) / cp.maximum(den_gpu, 1e-20)
        intercept = y_bar - slope * mu_gpu

        slope_cpu = cp.asnumpy(slope)
        intercept_cpu = cp.asnumpy(intercept)

        # Strength: pluggable. Default r²/SE² = slope² · maf(1-maf) · n · sxx / syy.
        strength_stats = {
            "slope":     slope_cpu,
            "intercept": intercept_cpu,
            "maf":       block_maf,
            "sxx":       block_sxx,
            "syy":       syy,
            "n":         n,
            "mu_x":      block_mu,
        }
        strength_cpu = _strength_callable(strength_stats)
        strength_cpu = np.asarray(strength_cpu, dtype=np.float64)

        compute_s = time.time() - t1
        total_compute_time += compute_s

        # Update top-K buffers — assign each variant to its window
        for i in range(take):
            variant_pos = base + i
            window_idx = min(variant_pos // window_size, n_windows - 1)

            topk_buffers[window_idx].update(
                gidx_batch=block_gidx[i:i + 1],
                genotypes_gpu=X_gpu[:, i:i + 1],
                mu_batch=block_mu[i:i + 1],
                den_batch=block_den[i:i + 1],
                alpha_batch=intercept_cpu[i:i + 1],
                beta_batch=slope_cpu[i:i + 1],
                strength_batch=strength_cpu[i:i + 1],
            )

        if show_block_times:
            min_strengths = [f"{b.min_strength:.4f}" for b in topk_buffers]
            log.info(
                "  block %d:%d (take=%d)  I/O=%.2fs  compute=%.2fs  window_mins=%s",
                base, end, take, io_s, compute_s, min_strengths,
            )

        # Release this block's GPU allocations back to the device before
        # reading the next block. Without this the CuPy memory pool retains
        # every block (~13 GB at 400K samples × 8192 SNPs × f32) and OOMs by
        # block 5-6 on an 80 GB GPU.
        del X_gpu, slope, intercept, x_c, mu_gpu, den_gpu
        cp.get_default_memory_pool().free_all_blocks()
        X_gpu = mu_gpu = den_gpu = x_c = slope = intercept = None

        base = end

    stream_time = time.time() - t_stream
    log.info("[PASS 1+2] Streaming completed in %.1fs", stream_time)
    log.info("  I/O time: %.1fs", total_io_time)
    log.info("  Compute time: %.1fs", total_compute_time)

    total_in_buffers = sum(b.count for b in topk_buffers)
    log.info(
        "  Total in buffers: %d SNPs across %d windows", total_in_buffers, n_windows,
    )

    reader.close()

    # Free leftover GPU arrays from the streaming loop
    if X_gpu is not None:
        del X_gpu, mu_gpu, den_gpu, x_c, slope, intercept
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    # ============================================================
    # COMBINE BUFFERS AND RUN LASSO
    # ============================================================
    log.info("[COMBINE] Merging %d window buffers...", n_windows)

    all_genotypes = []
    all_stats = []

    for i, buf in enumerate(topk_buffers):
        geno, stats = buf.get_sorted()
        if geno is not None:
            all_genotypes.append(geno)
            stats["window"] = i + 1
            all_stats.append(stats)

    if not all_stats:
        log.error("No SNPs in any window buffer — returning empty DataFrame")
        return pd.DataFrame(columns=[
            "gidx", "mu_x", "den", "alpha", "beta", "strength",
            "window", "lasso_weight", "final_coef_corrected",
        ])

    genotypes_gpu = cp.concatenate(all_genotypes, axis=1)
    stats_df = pd.concat(all_stats, ignore_index=True)

    del all_genotypes, topk_buffers
    gc.collect()
    cp.cuda.Stream.null.synchronize()
    cp.get_default_memory_pool().free_all_blocks()

    log.info(
        "  Combined: %d SNPs from %d windows", genotypes_gpu.shape[1], n_windows,
    )
    log.info("[LASSO] Running GPU LASSO on %d SNPs...", genotypes_gpu.shape[1])
    log.info("  Genotypes shape: %s", genotypes_gpu.shape)

    # Alpha-calibration hook: the I/O + strength + TopK selection above is
    # alpha-independent, so an alpha sweep can stop here and re-run only the
    # (cheap) LOO build + positive-LASSO via build_loo_and_run_lasso per trial.
    if return_candidates:
        return genotypes_gpu, stats_df, y_gpu

    result = build_loo_and_run_lasso(
        genotypes_gpu, stats_df, y_gpu,
        alpha_lasso=alpha, device=device,
    )

    # ============================================================
    # ASSEMBLE OUTPUT
    # ============================================================
    selected_mask = result["theta"] > 0
    output_df = stats_df[selected_mask].copy()
    output_df["lasso_weight"] = result["theta"][selected_mask]
    output_df["final_coef_corrected"] = result["final_slopes"][selected_mask]

    if output is not None:
        out_path = Path(output)
        if output_suffix:
            out_path = out_path.with_name(out_path.stem + output_suffix + out_path.suffix)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        output_df.to_feather(str(out_path))
        log.info("Saved %d selected SNPs to: %s", len(output_df), out_path)

        if save_npy:
            selected_indices = np.where(selected_mask)[0]
            selected_genotypes = cp.asnumpy(genotypes_gpu[:, selected_indices])
            target_npy_dir = Path(npy_dir) if npy_dir is not None else out_path.parent
            target_npy_dir.mkdir(parents=True, exist_ok=True)
            geno_file = target_npy_dir / f"chr{chr_label}_selected_genotypes.npy"
            np.save(str(geno_file), selected_genotypes)
            log.info("Saved selected genotypes to: %s", geno_file)

    total_time = time.time() - total_start
    log.info("=" * 70)
    log.info("CUGEN FUSED STEP 1+2 COMPLETE - WINDOWED (Chr %s)", chr_label)
    log.info("=" * 70)
    log.info("  Total time: %.1fs", total_time)
    log.info(
        "  I/O time: %.1fs (%.1f%%)",
        total_io_time, 100 * total_io_time / max(total_time, 1e-9),
    )
    log.info(
        "  Compute time: %.1fs (%.1f%%)",
        total_compute_time, 100 * total_compute_time / max(total_time, 1e-9),
    )
    log.info("  SNPs processed: %d", kept)
    log.info("  Windows: %d", n_windows)
    log.info(
        "  Top-K kept: %d (from %d windows)", total_in_buffers, n_windows,
    )
    log.info("  Selected by LASSO: %d", result["n_selected"])
    log.info("  R2: %.6f", result["r2"])

    return output_df


__all__ = [
    "screen_chromosome",
    "gpu_positive_lasso_fista",
    "build_loo_and_run_lasso",
    "TopKBuffer",
    "choose_batch_size",
]
