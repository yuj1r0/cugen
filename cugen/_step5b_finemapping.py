#!/usr/bin/env python3
"""
Step 5b: GPU fine-mapping — UltraSuSiE (Tier 1) + UltraMAP (Tier 2).

Tier 1 — UltraSuSiE: Fast genome-wide in-sample Bayesian fine-mapping.
    GPU-native SuSiE-RSS on exact in-sample LD from cugen genotypes.
    Produces per-variant PIPs and 95% credible sets.

Tier 2 — UltraMAP: Targeted dual-engine consensus fine-mapping.
    Adds LASSO CPSS stability selection, cross-engine convergence,
    and computes product-formula pseudo-PIPs (pPIPs) over the
    consensus credible set (cCS = SuSiE CS ∩ LASSO stability set).
    Classifies each locus: Resolved / Narrowed / Unresolved.

Both tiers use exact in-sample LD computed on GPU from cugen genotypes,
eliminating LD-mismatch artifacts that plague reference-panel-based fine-mapping.

Usage (Tier 1 — genome-wide UltraSuSiE):
    python step5b_finemapping.py --tier 1 --cache-stats \
        --loci results/loci_definition.tsv \
        --loco-pred results_v5_1stream/loco_predictions.npz \
        --cugen-dir /path/to/cugen_dir \
        --annotation gidx_annotation.feather \
        --output-dir results_ultrasusie_unified/

Usage (Tier 2 — targeted UltraMAP with cached stats):
    python step5b_finemapping.py --tier 2 --load-stats \
        --loci results/loci_definition.tsv \
        --loco-pred results_v5_1stream/loco_predictions.npz \
        --cugen-dir /path/to/cugen_dir \
        --annotation gidx_annotation.feather \
        --output-dir results_ultramap_unified/
"""
import argparse
import gc
import numpy as np
import cupy as cp
import pandas as pd
import time
import os
import multiprocessing as mp

from .io import CugenReader


# ---------------------------------------------------------------------------
# Track A: Elastic-FISTA on sufficient statistics
# ---------------------------------------------------------------------------

def elastic_fista_sufficient_stats(XtX, Xty, n, alpha, ridge_alpha=1e-4,
                                   max_iter=500, tol=1e-6, weights=None):
    """
    FISTA LASSO on sufficient statistics with Elastic Net regularization.

    The ridge term (ridge_alpha * I added to XtX) makes the objective strictly
    convex, giving the "grouping effect" — correlated variants share signal
    instead of vote-splitting.

    Args:
        XtX: (p, p) CuPy array — X'X / n (correlation-scale)
        Xty: (p,) CuPy array — X'y / n
        n: number of samples
        alpha: L1 penalty (LASSO)
        ridge_alpha: L2 penalty for Elastic Net (added to diagonal)
        max_iter: maximum iterations
        tol: convergence tolerance
        weights: (p,) CuPy array of per-variable L1 penalty weights, or None.
                 If provided, penalty for variable j = alpha * weights[j] * |theta_j|.
                 Higher weight = stronger penalty (harder to select).

    Returns:
        theta: (p,) CuPy array of coefficients
        n_active: number of non-zero coefficients
    """
    p = XtX.shape[0]

    # Add ridge to XtX for strict convexity
    XtX_reg = XtX.copy()
    if ridge_alpha > 0:
        XtX_reg[cp.arange(p), cp.arange(p)] += ridge_alpha

    # Lipschitz constant via power iteration
    v = cp.random.randn(p, dtype=cp.float32)
    v /= cp.linalg.norm(v)
    for _ in range(20):
        v_new = XtX_reg @ v
        v_new /= cp.linalg.norm(v_new)
        v = v_new
    L = float(cp.dot(v, XtX_reg @ v) / cp.dot(v, v))
    L = max(L, 1e-10)

    step = 0.95 / L
    # Per-variable threshold: scalar when uniform, (p,) vector when weighted
    if weights is not None:
        threshold = step * alpha * weights  # (p,) vector — broadcasts in soft-thresh
    else:
        threshold = step * alpha  # scalar

    theta = cp.zeros(p, dtype=cp.float32)
    theta_old = theta.copy()
    t = 1.0
    z = theta.copy()

    for k in range(max_iter):
        theta_old[:] = theta
        grad = XtX_reg @ z - Xty
        theta_new = z - step * grad
        theta = cp.sign(theta_new) * cp.maximum(cp.abs(theta_new) - threshold, 0)

        t_new = (1.0 + float(cp.sqrt(1.0 + 4.0 * t * t))) / 2.0
        z = theta + ((t - 1.0) / t_new) * (theta - theta_old)
        t = t_new

        if k % 50 == 0 and k > 0:
            diff = float(cp.linalg.norm(theta - theta_old))
            if diff < tol:
                break

    n_active = int(cp.sum(theta != 0))
    return theta, n_active


def compute_lambda_grid(Xty, n_lambda=20, lambda_min_ratio=0.01, weights=None):
    """
    Compute log-spaced lambda grid from lambda_max down.

    For unweighted LASSO: lambda_max = max(|Xty|). For adaptive LASSO with
    per-variable weights, the effective penalty for variant j at alpha is
    alpha * w_j, so the correct lambda_max (smallest alpha where all coefs
    are zero) is max(|Xty| / w) — otherwise the top-z variant is activated
    instantly at nominal lambda_max and the grid's high end is wasted in the
    saturated regime.

    Returns:
        (n_lambda,) numpy array of lambda values (descending)
    """
    abs_Xty = cp.abs(Xty)
    if weights is not None:
        # Effective |Xty/w| drives the active-set boundary under adaptive LASSO
        abs_Xty = abs_Xty / cp.maximum(weights, 1e-3)
    lambda_max = float(cp.max(abs_Xty))
    if lambda_max < 1e-12:
        return np.logspace(-6, -2, n_lambda)
    lambda_min = lambda_max * lambda_min_ratio
    return np.logspace(np.log10(lambda_max), np.log10(lambda_min), n_lambda)


# ---------------------------------------------------------------------------
# Low-memory helpers: compute sufficient stats via GPU streaming
# ---------------------------------------------------------------------------

def read_indices_centered_to_cpu(reader, local_indices, chunk_size=5000):
    """
    Read and center genotypes to CPU in GPU-chunks (for low-memory mode).

    Reads variants in chunks to GPU for 2-bit unpack, centers using column
    means, then copies to CPU. Never holds full X on GPU.

    Args:
        reader: CugenReader instance
        local_indices: numpy array of cugen-local variant indices
        chunk_size: number of variants per GPU chunk (5000 → ~6.7 GB)

    Returns:
        X_cpu: (n_samples, n_variants) numpy float32 — centered genotypes on CPU
    """
    n = reader.n_samples
    p = len(local_indices)
    X_cpu = np.empty((n, p), dtype=np.float32)

    for start in range(0, p, chunk_size):
        end = min(start + chunk_size, p)
        chunk_indices = local_indices[start:end]
        X_chunk_gpu = reader.read_indices_to_gpu(chunk_indices)
        mu = cp.mean(X_chunk_gpu, axis=0)
        X_chunk_gpu -= mu[None, :]
        X_cpu[:, start:end] = cp.asnumpy(X_chunk_gpu)
        del X_chunk_gpu, mu
        cp.get_default_memory_pool().free_all_blocks()

    return X_cpu


def compute_split_XtX_batched(X_cpu, y_cpu, indices, batch_size=8192):
    """
    Compute XtX/n and Xty/n for a sample subset by streaming batches to GPU.

    Peak GPU: batch_size × p × 4 bytes + p × p × 4 bytes.
    For p=3000, batch=8192: ~134 MB. For p=18000: ~2 GB.

    Args:
        X_cpu: (n_total, p) numpy float32 — centered genotypes on CPU
        y_cpu: (n_total,) numpy float32 — centered phenotype on CPU
        indices: (n_sub,) numpy int array — sample indices for this subset
        batch_size: number of samples per GPU batch

    Returns:
        XtX: (p, p) CuPy array on GPU — X_sub'X_sub / n_sub
        Xty: (p,) CuPy array on GPU — X_sub'y_sub / n_sub
    """
    p = X_cpu.shape[1]
    n_sub = len(indices)
    XtX = cp.zeros((p, p), dtype=cp.float32)
    Xty = cp.zeros(p, dtype=cp.float32)

    for start in range(0, n_sub, batch_size):
        end = min(start + batch_size, n_sub)
        batch_idx = indices[start:end]
        X_batch = cp.asarray(X_cpu[batch_idx])
        y_batch = cp.asarray(y_cpu[batch_idx])
        XtX += X_batch.T @ X_batch
        Xty += X_batch.T @ y_batch
        del X_batch, y_batch

    XtX /= n_sub
    Xty /= n_sub
    cp.get_default_memory_pool().free_all_blocks()
    return XtX, Xty


def stability_selection_cpss_lowmem(X_cpu, y_cpu, n_pairs=50, n_lambda=20,
                                    ridge_alpha=1e-4, seed=42, weights=None,
                                    lambda_min_ratio=0.01,
                                    batch_size=8192, verbose=True):
    """
    Low-memory CPSS: X stays on CPU, stream batches to GPU for XtX.

    Peak GPU memory: max(batch_size × p × 4, 2 × p × p × 4) bytes.
    For p=3000, batch=8192: ~134 MB. For p=18000: ~2.6 GB.

    Same algorithm and outputs as stability_selection_cpss, but dramatically
    lower GPU memory at the cost of ~10-20% more time (streaming overhead).

    Args:
        X_cpu: (n, p) numpy float32 — centered genotypes on CPU
        y_cpu: (n,) numpy float32 — centered phenotype on CPU
        n_pairs: number of complementary half-splits
        n_lambda: number of lambda values
        ridge_alpha: Elastic Net L2 penalty
        seed: random seed
        weights: (p,) CuPy array of per-variable L1 penalty weights, or None
        batch_size: samples per GPU batch for XtX computation
        verbose: print progress

    Returns:
        sel_freq: (p,) numpy array — selection frequency in [0, 1]
        avg_beta: (p,) numpy array — average coefficient magnitude
        unique_models: list of frozensets of active indices
        per_lambda_freq: (n_lambda, p) numpy array — per-lambda selection frequency
    """
    n, p = X_cpu.shape

    # Precompute full-sample sufficient statistics via batched streaming
    if verbose:
        print(f"    Computing XtX_full via batched streaming (batch={batch_size})...")
    XtX_full, Xty_full = compute_split_XtX_batched(
        X_cpu, y_cpu, np.arange(n), batch_size=batch_size)

    lambdas = compute_lambda_grid(Xty_full, n_lambda=n_lambda,
                                  lambda_min_ratio=lambda_min_ratio,
                                  weights=weights)

    # Accumulators
    total_selections = np.zeros(p, dtype=np.float64)
    total_beta = np.zeros(p, dtype=np.float64)
    per_lambda_counts = np.zeros((n_lambda, p), dtype=np.float64)
    unique_models = set()
    total_runs = 0

    rng = np.random.RandomState(seed)

    t0 = time.time()
    for pair_idx in range(n_pairs):
        perm = rng.permutation(n)
        n_half = n // 2
        idx_a = perm[:n_half]

        # Compute XtX_a via batched streaming (only half A)
        XtX_a, Xty_a = compute_split_XtX_batched(
            X_cpu, y_cpu, idx_a, batch_size=batch_size)

        # Half B via complementary trick (free!)
        n_used = 2 * n_half
        XtX_b = (XtX_full * n_used - XtX_a * n_half) / n_half
        Xty_b = (Xty_full * n_used - Xty_a * n_half) / n_half

        # Run LASSO at each lambda on both halves
        for li, lam in enumerate(lambdas):
            for XtX_sub, Xty_sub in [(XtX_a, Xty_a), (XtX_b, Xty_b)]:
                theta, n_act = elastic_fista_sufficient_stats(
                    XtX_sub, Xty_sub, n_half, alpha=lam,
                    ridge_alpha=ridge_alpha, max_iter=300, tol=1e-5,
                    weights=weights)

                active = cp.asnumpy(theta != 0)
                total_selections += active
                total_beta += np.abs(cp.asnumpy(theta))
                per_lambda_counts[li] += active
                total_runs += 1

                if n_act > 0 and n_act <= max(100, p // 10):
                    active_set = frozenset(np.where(active)[0].tolist())
                    unique_models.add(active_set)

        del XtX_a, Xty_a, XtX_b, Xty_b
        cp.get_default_memory_pool().free_all_blocks()

        if verbose and (pair_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    CPSS pair {pair_idx+1}/{n_pairs} "
                  f"({elapsed:.1f}s, {len(unique_models)} unique models)")

    sel_freq = total_selections / total_runs
    avg_beta = total_beta / total_runs
    per_lambda_freq = per_lambda_counts / (2 * n_pairs)

    del XtX_full, Xty_full
    cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        n_selected = np.sum(sel_freq > 0.5)
        print(f"    CPSS complete: {n_selected} variants with freq > 0.5, "
              f"{len(unique_models)} unique models")

    return sel_freq, avg_beta, list(unique_models), per_lambda_freq


# ---------------------------------------------------------------------------
# Track A: Complementary Pairs Stability Selection (CPSS)
# ---------------------------------------------------------------------------

def stability_selection_cpss(X_gpu, y_gpu, n_pairs=50, n_lambda=20,
                             ridge_alpha=1e-4, seed=42, weights=None,
                             lambda_min_ratio=0.01,
                             verbose=True):
    """
    Complementary Pairs Stability Selection (Shah & Samworth 2013).

    For each of n_pairs random splits:
      - Split samples into complementary halves A and B
      - Compute XtX_A on half A; get XtX_B via complementary trick (free!)
      - Run elastic-FISTA at each lambda on both halves
      - Track selection frequency and unique active sets (models)

    Args:
        X_gpu: (n, p) CuPy array — centered genotypes
        y_gpu: (n,) CuPy array — centered phenotype
        n_pairs: number of complementary half-splits
        n_lambda: number of lambda values
        ridge_alpha: Elastic Net L2 penalty
        seed: random seed
        weights: (p,) CuPy array of per-variable L1 penalty weights, or None
        verbose: print progress

    Returns:
        sel_freq: (p,) numpy array — selection frequency in [0, 1]
        avg_beta: (p,) numpy array — average coefficient magnitude
        unique_models: list of frozensets of active indices
        per_lambda_freq: (n_lambda, p) numpy array — per-lambda selection frequency
    """
    n, p = X_gpu.shape

    # Precompute full-sample sufficient statistics
    XtX_full = cp.dot(X_gpu.T, X_gpu) / n
    Xty_full = cp.dot(X_gpu.T, y_gpu) / n

    lambdas = compute_lambda_grid(Xty_full, n_lambda=n_lambda,
                                  lambda_min_ratio=lambda_min_ratio,
                                  weights=weights)

    # Accumulators
    total_selections = np.zeros(p, dtype=np.float64)
    total_beta = np.zeros(p, dtype=np.float64)
    per_lambda_counts = np.zeros((n_lambda, p), dtype=np.float64)
    unique_models = set()
    total_runs = 0

    rng = np.random.RandomState(seed)

    t0 = time.time()
    for pair_idx in range(n_pairs):
        # Random complementary split
        perm = rng.permutation(n)
        n_half = n // 2
        idx_a = perm[:n_half]

        # Compute XtX for half A
        X_a = X_gpu[idx_a]
        y_a = y_gpu[idx_a]
        XtX_a = cp.dot(X_a.T, X_a) / n_half
        Xty_a = cp.dot(X_a.T, y_a) / n_half
        del X_a, y_a

        # Half B via complementary trick: XtX_b from XtX_full - XtX_a
        # Use n_used = 2*n_half (drops 1 sample if n is odd — negligible bias)
        n_used = 2 * n_half
        XtX_b = (XtX_full * n_used - XtX_a * n_half) / n_half
        Xty_b = (Xty_full * n_used - Xty_a * n_half) / n_half

        # Run LASSO at each lambda on both halves
        for li, lam in enumerate(lambdas):
            for XtX_sub, Xty_sub in [(XtX_a, Xty_a), (XtX_b, Xty_b)]:
                theta, n_act = elastic_fista_sufficient_stats(
                    XtX_sub, Xty_sub, n_half, alpha=lam,
                    ridge_alpha=ridge_alpha, max_iter=300, tol=1e-5,
                    weights=weights)

                active = cp.asnumpy(theta != 0)
                total_selections += active
                total_beta += np.abs(cp.asnumpy(theta))
                per_lambda_counts[li] += active
                total_runs += 1

                # Track unique model (active set)
                if n_act > 0 and n_act <= max(100, p // 10):
                    active_set = frozenset(np.where(active)[0].tolist())
                    unique_models.add(active_set)

        del XtX_a, Xty_a, XtX_b, Xty_b
        cp.get_default_memory_pool().free_all_blocks()

        if verbose and (pair_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    CPSS pair {pair_idx+1}/{n_pairs} "
                  f"({elapsed:.1f}s, {len(unique_models)} unique models)")

    # Normalize
    sel_freq = total_selections / total_runs
    avg_beta = total_beta / total_runs
    per_lambda_freq = per_lambda_counts / (2 * n_pairs)  # 2 halves per pair

    del XtX_full, Xty_full
    cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        n_selected = np.sum(sel_freq > 0.5)
        print(f"    CPSS complete: {n_selected} variants with freq > 0.5, "
              f"{len(unique_models)} unique models")

    return sel_freq, avg_beta, list(unique_models), per_lambda_freq


def stability_selection_cpss_from_stats(XtX_full, Xty_full, var_y, n_samples,
                                        n_pairs=50, n_lambda=20,
                                        ridge_alpha=1e-4, seed=42, weights=None,
                                        lambda_min_ratio=0.01,
                                        verbose=True):
    """
    CPSS from cached sufficient statistics (no individual-level data).

    Approximates subsampling by adding scaled random noise to XtX/Xty.
    Each pair creates two perturbed copies: XtX_pert = XtX + N(0, sigma²/n),
    simulating the sampling variance of a half-sample estimator.

    This is an approximation — use when individual data is unavailable
    (e.g., Tier 2 with --load-stats from Tier 1 cache).

    Returns same interface as stability_selection_cpss.
    """
    p = XtX_full.shape[0]
    n_half = n_samples // 2
    lambdas = compute_lambda_grid(Xty_full, n_lambda=n_lambda,
                                  lambda_min_ratio=lambda_min_ratio,
                                  weights=weights)

    # Sampling noise scale: var(XtX_half) ≈ (2/n) * XtX_full
    noise_scale_XtX = float(cp.sqrt(2.0 / n_samples))
    noise_scale_Xty = float(cp.sqrt(2.0 / n_samples * var_y))

    total_selections = np.zeros(p, dtype=np.float64)
    total_beta = np.zeros(p, dtype=np.float64)
    per_lambda_counts = np.zeros((n_lambda, p), dtype=np.float64)
    unique_models = set()
    total_runs = 0

    rng_cp = cp.random.RandomState(seed)
    t0 = time.time()

    for pair_idx in range(n_pairs):
        # Generate symmetric noise: A gets +noise, B gets -noise
        noise_XtX = rng_cp.randn(p, p, dtype=cp.float32) * noise_scale_XtX
        noise_XtX = (noise_XtX + noise_XtX.T) / 2  # Symmetrize
        noise_Xty = rng_cp.randn(p, dtype=cp.float32) * noise_scale_Xty

        XtX_a = XtX_full + noise_XtX
        Xty_a = Xty_full + noise_Xty
        XtX_b = XtX_full - noise_XtX
        Xty_b = Xty_full - noise_Xty
        del noise_XtX, noise_Xty

        for li, lam in enumerate(lambdas):
            for XtX_sub, Xty_sub in [(XtX_a, Xty_a), (XtX_b, Xty_b)]:
                theta, n_act = elastic_fista_sufficient_stats(
                    XtX_sub, Xty_sub, n_half, alpha=lam,
                    ridge_alpha=ridge_alpha, max_iter=300, tol=1e-5,
                    weights=weights)

                active = cp.asnumpy(theta != 0)
                total_selections += active
                total_beta += np.abs(cp.asnumpy(theta))
                per_lambda_counts[li] += active
                total_runs += 1

                if n_act > 0 and n_act <= max(100, p // 10):
                    active_set = frozenset(np.where(active)[0].tolist())
                    unique_models.add(active_set)

        del XtX_a, Xty_a, XtX_b, Xty_b
        cp.get_default_memory_pool().free_all_blocks()

        if verbose and (pair_idx + 1) % 10 == 0:
            elapsed = time.time() - t0
            print(f"    CPSS-from-stats pair {pair_idx+1}/{n_pairs} "
                  f"({elapsed:.1f}s, {len(unique_models)} unique models)")

    sel_freq = total_selections / total_runs
    avg_beta = total_beta / total_runs
    per_lambda_freq = per_lambda_counts / (2 * n_pairs)

    if verbose:
        n_selected = np.sum(sel_freq > 0.5)
        print(f"    CPSS-from-stats complete: {n_selected} variants with freq > 0.5, "
              f"{len(unique_models)} unique models")

    return sel_freq, avg_beta, list(unique_models), per_lambda_freq


# ---------------------------------------------------------------------------
# Track A: Bayesian Model Averaging over LASSO-proposed models
# ---------------------------------------------------------------------------

def compute_bma_pips(unique_models, XtX_full, Xty_full, var_y, n_samples,
                     max_models=5000, verbose=True):
    """
    Bayesian Model Averaging over LASSO-proposed models.

    For each unique active set from CPSS:
      1. Subset XtX/Xty to active variables
      2. OLS fit: beta = (XtX_sub)^{-1} Xty_sub
      3. Compute BIC = n*log(RSS/n) + k*log(n)
      4. Model posterior ∝ exp(-0.5 * delta_BIC)
      5. PIP_j = sum of posterior probabilities of models containing j

    Args:
        unique_models: list of frozensets of active indices
        XtX_full: (p, p) CuPy array — X'X/n
        Xty_full: (p,) CuPy array — X'y/n
        var_y: float — variance of y
        n_samples: int
        max_models: maximum models to evaluate
        verbose: print progress

    Returns:
        pips: (p,) numpy array — posterior inclusion probabilities
    """
    p = XtX_full.shape[0]
    n = n_samples

    # Filter valid models
    valid_models = [m for m in unique_models if 0 < len(m) <= 100]
    if len(valid_models) > max_models:
        # Keep models with most variants (tend to be more informative)
        valid_models.sort(key=len, reverse=True)
        valid_models = valid_models[:max_models]

    if len(valid_models) == 0:
        if verbose:
            print("    BMA: no valid models, returning zeros")
        return np.zeros(p, dtype=np.float64)

    if verbose:
        print(f"    BMA: evaluating {len(valid_models)} models")

    # Move to numpy for BIC computation (small matrices)
    XtX_np = cp.asnumpy(XtX_full)
    Xty_np = cp.asnumpy(Xty_full)

    bic_values = []
    model_list = []

    for model in valid_models:
        idx = sorted(model)
        k = len(idx)

        # Subset
        XtX_sub = XtX_np[np.ix_(idx, idx)]
        Xty_sub = Xty_np[idx]

        # OLS: beta = (XtX_sub)^{-1} Xty_sub
        try:
            beta = np.linalg.solve(XtX_sub, Xty_sub)
        except np.linalg.LinAlgError:
            continue

        # RSS/n = var_y - beta' Xty_sub (since X and y are centered)
        rss_over_n = var_y - float(beta @ Xty_sub)
        if rss_over_n <= 0:
            rss_over_n = 1e-10  # Numerical guard

        # BIC = n * log(RSS/n) + k * log(n)
        bic = n * np.log(rss_over_n) + k * np.log(n)
        bic_values.append(bic)
        model_list.append(idx)

    if len(bic_values) == 0:
        if verbose:
            print("    BMA: all models failed OLS, returning zeros")
        return np.zeros(p, dtype=np.float64)

    bic_values = np.array(bic_values)

    # Model posterior: exp(-0.5 * delta_BIC)
    bic_min = bic_values.min()
    log_weights = -0.5 * (bic_values - bic_min)

    # Numerical stability: shift so max is 0
    log_weights -= log_weights.max()
    weights = np.exp(log_weights)
    weights /= weights.sum()

    # PIP: sum of posterior probabilities of models containing variant j
    pips = np.zeros(p, dtype=np.float64)
    for w, idx in zip(weights, model_list):
        pips[idx] += w

    if verbose:
        n_high = np.sum(pips > 0.5)
        best_k = len(model_list[np.argmax(weights)])
        print(f"    BMA: {n_high} variants with PIP > 0.5, "
              f"best model has {best_k} variants")

    return pips


# ---------------------------------------------------------------------------
# Track B: GPU SuSiE-RSS
# ---------------------------------------------------------------------------

def gpu_susie_rss(XtX, Xty, var_y, n_samples, L=10, max_iter=100,
                  prior_var=50.0, tol=1e-4, prior_pi=None, verbose=True):
    """
    GPU-native SuSiE-RSS (Sum of Single Effects, RSS likelihood).

    Implements the Iterative Bayesian Stepwise Selection (IBSS) algorithm
    on summary statistics (R, z-scores) derived from XtX and Xty.

    Args:
        XtX: (p, p) CuPy array — X'X/n (covariance-scale, centered X)
        Xty: (p,) CuPy array — X'y/n
        var_y: float — var(y)
        n_samples: int
        L: number of single effects
        max_iter: maximum IBSS iterations
        prior_var: prior variance for each effect (in z-score scale)
        tol: ELBO convergence tolerance
        prior_pi: (p,) CuPy array — prior inclusion probabilities, or None
                  for uniform. Used in cross-engine convergence to inject
                  LASSO selection frequencies as SuSiE priors.
        verbose: print progress

    Returns:
        pips: (p,) numpy array — posterior inclusion probabilities
        cs_list: list of dicts with credible set info
    """
    p = XtX.shape[0]
    n = n_samples

    # Convert to correlation matrix R and z-scores
    # diag(XtX) = var(X_j) after centering
    diag_XtX = cp.diag(XtX).copy()
    diag_XtX = cp.maximum(diag_XtX, cp.float32(1e-10))
    sd = cp.sqrt(diag_XtX)

    # R = D^{-1} XtX D^{-1} where D = diag(sd)
    R = XtX / (sd[:, None] * sd[None, :])
    # Clip R diagonal to exactly 1
    R[cp.arange(p), cp.arange(p)] = 1.0

    # z = sqrt(n) * beta_hat * sd_x / sd_y, where beta_hat = Xty / diag(XtX)
    beta_hat = Xty / diag_XtX
    sd_y = cp.float32(np.sqrt(var_y))
    z = cp.sqrt(cp.float32(n)) * beta_hat * sd / sd_y

    # Add small ridge to R diagonal for numerical stability
    R[cp.arange(p), cp.arange(p)] += cp.float32(1e-4)

    # Initialize SuSiE parameters
    if prior_pi is not None:
        # Non-uniform prior: initialize alpha from prior_pi
        log_prior = cp.log(cp.maximum(prior_pi, cp.float32(1e-10)))
        alpha = cp.tile(prior_pi, (L, 1))  # (L, p) — each effect starts at prior
    else:
        log_prior = cp.float32(0.0)  # Uniform prior (cancels in softmax)
        alpha = cp.ones((L, p), dtype=cp.float32) / p  # Uniform initialization
    mu = cp.zeros((L, p), dtype=cp.float32)         # Posterior means
    mu2 = cp.zeros((L, p), dtype=cp.float32)        # Posterior second moments

    # IBSS iterations
    for iteration in range(max_iter):
        alpha_old = alpha.copy()

        for l_idx in range(L):
            # Compute residual z-scores (remove all other effects)
            # r_l = z - sum_{l' != l} R @ (alpha_{l'} * mu_{l'})
            r_l = z.copy()
            for l2 in range(L):
                if l2 != l_idx:
                    r_l -= R @ (alpha[l2] * mu[l2])

            # Single effect regression (SER)
            # log BF_j = 0.5 * log(1/(1+prior_var)) + 0.5 * prior_var/(1+prior_var) * r_j^2
            sigma2 = 1.0 + prior_var
            log_bf = (0.5 * cp.log(cp.float32(1.0 / sigma2))
                      + 0.5 * (prior_var / sigma2) * r_l * r_l)

            # Add log prior (non-uniform when prior_pi provided)
            log_bf = log_bf + log_prior

            # Posterior inclusion probability (softmax of log BF + log prior)
            log_bf_max = cp.max(log_bf)
            alpha[l_idx] = cp.exp(log_bf - log_bf_max)
            alpha_sum = cp.sum(alpha[l_idx])
            if float(alpha_sum) > 0:
                alpha[l_idx] /= alpha_sum
            else:
                alpha[l_idx] = cp.float32(1.0 / p)

            # Posterior mean and second moment
            post_var = prior_var / sigma2
            mu[l_idx] = post_var * r_l
            mu2[l_idx] = post_var + mu[l_idx] ** 2

        # Check convergence
        diff = float(cp.max(cp.abs(alpha - alpha_old)))
        if diff < tol:
            if verbose:
                print(f"    SuSiE converged at iteration {iteration+1} "
                      f"(max alpha change: {diff:.2e})")
            break

    if verbose and iteration == max_iter - 1:
        print(f"    SuSiE reached max iterations ({max_iter}), "
              f"final change: {diff:.2e}")

    # Compute PIPs: PIP_j = 1 - prod_l(1 - alpha_lj)
    alpha_np = cp.asnumpy(alpha)
    pips = 1.0 - np.prod(1.0 - alpha_np, axis=0)

    # Build credible sets for each effect
    cs_list = []
    for l_idx in range(L):
        a = alpha_np[l_idx]
        # Check if this effect is "real" (not uniformly spread)
        max_pip = a.max()
        if max_pip < 0.05:
            continue  # Skip uninformative effects

        # Build 95% credible set
        sorted_idx = np.argsort(-a)
        cumsum = np.cumsum(a[sorted_idx])
        cs_size = int(np.searchsorted(cumsum, 0.95)) + 1
        cs_indices = sorted_idx[:cs_size].tolist()

        cs_list.append({
            'effect_idx': l_idx,
            'cs_indices': cs_indices,
            'cs_size': cs_size,
            'max_pip': float(max_pip),
            'cs_coverage': float(cumsum[cs_size - 1]) if cs_size <= len(cumsum) else 1.0,
        })

    # Cleanup
    del R, z, alpha, mu, mu2, r_l, beta_hat, sd, diag_XtX, log_bf
    cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        n_high = np.sum(pips > 0.5)
        print(f"    SuSiE: {n_high} variants with PIP > 0.5, "
              f"{len(cs_list)} credible sets")

    return pips, cs_list


# ---------------------------------------------------------------------------
# Variance explained
# ---------------------------------------------------------------------------

def compute_variance_explained(X_gpu, y_gpu, theta_gpu, maf):
    """
    Compute per-variant heritability contribution.

    Joint R² from LASSO prediction and per-variant h² = beta² × 2f(1-f).

    Returns:
        joint_r2: float — R² from the LASSO model
        h2_per_variant: (p,) numpy array
    """
    # Joint R²
    y_pred = X_gpu @ theta_gpu
    ss_res = float(cp.sum((y_gpu - y_pred) ** 2))
    ss_tot = float(cp.sum(y_gpu ** 2))  # y is already centered
    joint_r2 = 1.0 - ss_res / max(ss_tot, 1e-10)

    # Per-variant h² = beta² × 2*maf*(1-maf)
    beta_np = cp.asnumpy(theta_gpu)
    h2_per = beta_np ** 2 * 2.0 * maf * (1.0 - maf)

    return joint_r2, h2_per


def compute_variance_explained_from_stats(XtX, Xty, var_y, theta_gpu, maf):
    """
    Compute variance explained from sufficient statistics (no X needed).

    R² = 1 - (var_y - 2*theta'*Xty + theta'*XtX*theta) / var_y

    Exact when X and y are centered, which they are in our pipeline.

    Args:
        XtX: (p, p) CuPy array — X'X/n
        Xty: (p,) CuPy array — X'y/n
        var_y: float — var(y)
        theta_gpu: (p,) CuPy array — LASSO coefficients
        maf: (p,) numpy array — minor allele frequencies

    Returns:
        joint_r2: float — R² from the LASSO model
        h2_per_variant: (p,) numpy array
    """
    rss_over_n = var_y - 2.0 * float(cp.dot(theta_gpu, Xty)) + float(theta_gpu @ XtX @ theta_gpu)
    rss_over_n = max(rss_over_n, 1e-10)
    joint_r2 = 1.0 - rss_over_n / max(var_y, 1e-10)

    beta_np = cp.asnumpy(theta_gpu)
    h2_per = beta_np ** 2 * 2.0 * maf * (1.0 - maf)

    return joint_r2, h2_per


# ---------------------------------------------------------------------------
# Cross-engine convergence
# ---------------------------------------------------------------------------

def cross_engine_convergence(XtX, Xty, var_y, n_samples,
                             n_pairs=50, n_lambda=20, ridge_alpha=1e-4,
                             susie_L=10, max_rounds=3,
                             discrepancy_threshold=0.3, prior_strength=0.5,
                             X_gpu=None, y_gpu=None,
                             X_cpu=None, y_cpu=None, batch_size=8192,
                             verbose=True):
    """
    Iterative cross-engine convergence between LASSO CPSS and SuSiE.

    Round 0: Independent runs (no priors).
    Round k (k=1..max_rounds):
      - SuSiE PIPs → adaptive LASSO weights (high PIP = low penalty)
      - Run weighted CPSS → BMA PIPs
      - LASSO sel_freq → SuSiE prior_pi (high freq = high prior)
      - Run SuSiE with prior → SuSiE PIPs
      - Check convergence: max|BMA_PIP - SuSiE_PIP| < threshold

    The prior_strength controls blending:
      0 = ignore other engine entirely
      1 = fully trust other engine's output

    Args:
        XtX, Xty, var_y, n_samples: sufficient statistics
        n_pairs, n_lambda, ridge_alpha: CPSS params
        susie_L: SuSiE effects
        max_rounds: max convergence iterations after round 0
        discrepancy_threshold: stop when max PIP discrepancy < this
        prior_strength: blending weight (0-1)
        X_gpu, y_gpu: CuPy arrays for normal-memory CPSS (mutually exclusive with X_cpu)
        X_cpu, y_cpu: numpy arrays for low-memory CPSS
        batch_size: for low-memory batched streaming
        verbose: print progress

    Returns:
        dict with all fine-mapping outputs + convergence history
    """
    p = XtX.shape[0]
    use_lowmem = X_cpu is not None
    history = []

    # --- Round 0: Independent runs ---
    if verbose:
        print(f"\n  --- Convergence Round 0: Independent runs ---")

    # Track A: CPSS
    t_a = time.time()
    if use_lowmem:
        sel_freq, avg_beta, unique_models, per_lambda_freq = \
            stability_selection_cpss_lowmem(
                X_cpu, y_cpu, n_pairs=n_pairs, n_lambda=n_lambda,
                ridge_alpha=ridge_alpha, batch_size=batch_size,
                verbose=verbose)
    else:
        sel_freq, avg_beta, unique_models, per_lambda_freq = \
            stability_selection_cpss(
                X_gpu, y_gpu, n_pairs=n_pairs, n_lambda=n_lambda,
                ridge_alpha=ridge_alpha, verbose=verbose)

    bma_pips = compute_bma_pips(
        unique_models, XtX, Xty, var_y, n_samples, verbose=verbose)
    if verbose:
        print(f"  Track A (round 0) completed in {time.time()-t_a:.1f}s")

    # Track B: SuSiE
    t_b = time.time()
    susie_pips, susie_cs_list = gpu_susie_rss(
        XtX, Xty, var_y, n_samples, L=susie_L, verbose=verbose)
    if verbose:
        print(f"  Track B (round 0) completed in {time.time()-t_b:.1f}s")

    # Check discrepancy
    max_disc = float(np.max(np.abs(bma_pips - susie_pips)))
    mean_disc = float(np.mean(np.abs(bma_pips - susie_pips)))
    n_high_lasso = int(np.sum(bma_pips > 0.5))
    n_high_susie = int(np.sum(susie_pips > 0.5))

    history.append({
        'round': 0,
        'max_discrepancy': max_disc,
        'mean_discrepancy': mean_disc,
        'n_pip50_lasso': n_high_lasso,
        'n_pip50_susie': n_high_susie,
        'prior_strength_used': 0.0,
    })

    if verbose:
        print(f"\n  Round 0 discrepancy: max={max_disc:.3f}, mean={mean_disc:.3f}")
        print(f"    PIP>0.5: LASSO={n_high_lasso}, SuSiE={n_high_susie}")

    # --- Check lead-variant agreement ---
    # Simple criterion: do both engines point to the same top variant?
    # If yes → skip convergence (saves ~4 min of 2000 LASSO refits).
    # If no  → run convergence to resolve disagreement.
    #
    # We also check if LASSO's lead is anywhere in a SuSiE CS (partial agree).
    # Future: could compute P(same causal) accounting for CS size / LD.

    lasso_lead = int(np.argmax(bma_pips))
    susie_lead = int(np.argmax(susie_pips))
    same_lead = (lasso_lead == susie_lead)

    # Check if LASSO lead is in any SuSiE CS
    susie_cs_members = set()
    for cs in susie_cs_list:
        for idx in cs['cs_indices']:
            susie_cs_members.add(idx)
    lasso_lead_in_cs = lasso_lead in susie_cs_members

    if verbose:
        print(f"    Lead variants: LASSO={lasso_lead} SuSiE={susie_lead} "
              f"{'SAME' if same_lead else 'DIFFERENT'}")
        if not same_lead:
            print(f"    LASSO lead in SuSiE CS: {lasso_lead_in_cs} "
                  f"(CS members: {len(susie_cs_members)})")

    skip_result = {
        'sel_freq': sel_freq, 'avg_beta': avg_beta,
        'unique_models': unique_models, 'per_lambda_freq': per_lambda_freq,
        'bma_pips': bma_pips, 'susie_pips': susie_pips,
        'susie_cs_list': susie_cs_list, 'history': history,
    }

    if same_lead:
        if verbose:
            print(f"  Same lead variant → skipping convergence")
        return skip_result

    if lasso_lead_in_cs:
        if verbose:
            print(f"  LASSO lead in SuSiE CS (partial agree) → skipping convergence")
        return skip_result

    if len(susie_cs_list) == 0:
        if verbose:
            print(f"  SuSiE found no credible sets → skipping convergence")
        return skip_result

    if verbose:
        print(f"  Lead variant DISAGREEMENT (LASSO lead not in any SuSiE CS) "
              f"→ running convergence")

    # --- Convergence rounds ---
    n_high_lasso_prev = n_high_lasso
    cs_agree_now = False
    for round_idx in range(1, max_rounds + 1):
        if verbose:
            print(f"\n  --- Convergence Round {round_idx}/{max_rounds} "
                  f"(strength={prior_strength:.2f}) ---")

        # SuSiE CS → adaptive LASSO weights (CS-based, NOT PIP-based)
        # Only boost variants that are IN a SuSiE credible set.
        # Boost scales with 1/cs_size: size-1 CS = strong boost, size-100 = weak.
        # Variants outside any CS keep weight=1.0 (default penalty).
        weights_np = np.ones(p, dtype=np.float32)
        for cs in susie_cs_list:
            cs_idx = cs['cs_indices']
            cs_sz = cs['cs_size']
            # Reduction: strength * (1/cs_size), capped so weight >= 0.1
            # Small CS: weight → 1 - strength ≈ 0.5 (strong boost to each member)
            # Large CS (100+): weight → 1 - strength/100 ≈ 0.995 (negligible boost)
            reduction = prior_strength / max(cs_sz, 1)
            for idx in cs_idx:
                weights_np[idx] = min(weights_np[idx],
                                      max(1.0 - reduction, 0.1))
        weights_gpu = cp.asarray(weights_np)

        n_boosted = int(np.sum(weights_np < 1.0))
        if verbose:
            print(f"    CS-based weights: {n_boosted} variants boosted "
                  f"(min weight={weights_np.min():.3f})")

        # Re-run weighted CPSS
        t_a = time.time()
        if use_lowmem:
            sel_freq, avg_beta, unique_models, per_lambda_freq = \
                stability_selection_cpss_lowmem(
                    X_cpu, y_cpu, n_pairs=n_pairs, n_lambda=n_lambda,
                    ridge_alpha=ridge_alpha, weights=weights_gpu,
                    batch_size=batch_size, seed=42 + round_idx,
                    verbose=verbose)
        else:
            sel_freq, avg_beta, unique_models, per_lambda_freq = \
                stability_selection_cpss(
                    X_gpu, y_gpu, n_pairs=n_pairs, n_lambda=n_lambda,
                    ridge_alpha=ridge_alpha, weights=weights_gpu,
                    seed=42 + round_idx, verbose=verbose)

        bma_pips = compute_bma_pips(
            unique_models, XtX, Xty, var_y, n_samples, verbose=verbose)
        if verbose:
            print(f"  Weighted CPSS (round {round_idx}) completed in {time.time()-t_a:.1f}s")

        # LASSO sel_freq → SuSiE prior_pi
        # High selection freq = high prior inclusion probability
        # pi_j = strength * freq_j + (1-strength)/p, normalized to sum=1
        prior_pi_np = prior_strength * sel_freq + (1.0 - prior_strength) / p
        prior_pi_np = prior_pi_np / prior_pi_np.sum()  # Normalize to sum=1
        prior_pi_gpu = cp.asarray(prior_pi_np.astype(np.float32))

        # Re-run SuSiE with informative prior
        t_b = time.time()
        susie_pips, susie_cs_list = gpu_susie_rss(
            XtX, Xty, var_y, n_samples, L=susie_L,
            prior_pi=prior_pi_gpu, verbose=verbose)
        if verbose:
            print(f"  SuSiE with prior (round {round_idx}) completed in {time.time()-t_b:.1f}s")

        # Re-check lead-variant agreement after this round
        lasso_lead_new = int(np.argmax(bma_pips))
        susie_lead_new = int(np.argmax(susie_pips))
        same_lead_now = (lasso_lead_new == susie_lead_new)

        susie_cs_members = set()
        for cs in susie_cs_list:
            for idx in cs['cs_indices']:
                susie_cs_members.add(idx)
        lasso_lead_in_cs_now = lasso_lead_new in susie_cs_members

        max_disc = float(np.max(np.abs(bma_pips - susie_pips)))
        mean_disc = float(np.mean(np.abs(bma_pips - susie_pips)))
        n_high_lasso = int(np.sum(bma_pips > 0.5))
        n_high_susie = int(np.sum(susie_pips > 0.5))

        history.append({
            'round': round_idx,
            'max_discrepancy': max_disc,
            'mean_discrepancy': mean_disc,
            'n_pip50_lasso': n_high_lasso,
            'n_pip50_susie': n_high_susie,
            'prior_strength_used': prior_strength,
            'same_lead': same_lead_now,
            'lasso_lead_in_cs': lasso_lead_in_cs_now,
        })

        cs_agree_now = same_lead_now or lasso_lead_in_cs_now

        if verbose:
            print(f"\n  Round {round_idx}: max_disc={max_disc:.3f}, "
                  f"PIP>0.5: LASSO={n_high_lasso} SuSiE={n_high_susie}")
            print(f"    Leads: LASSO={lasso_lead_new} SuSiE={susie_lead_new} "
                  f"{'SAME' if same_lead_now else 'DIFF'}, "
                  f"LASSO lead in CS: {lasso_lead_in_cs_now}")

        if cs_agree_now:
            if verbose:
                print(f"  Lead agreement reached at round {round_idx}!")
            break

        # Safety: if LASSO PIP>50% count is growing, stop (diverging)
        if round_idx > 1 and n_high_lasso > n_high_lasso_prev * 2:
            if verbose:
                print(f"  WARNING: LASSO PIP>50% growing ({n_high_lasso} vs "
                      f"prev {n_high_lasso_prev}), stopping to prevent divergence")
            break

        n_high_lasso_prev = n_high_lasso

    if verbose and not cs_agree_now:
        print(f"  Did not converge after {max_rounds} rounds")

    return {
        'sel_freq': sel_freq, 'avg_beta': avg_beta,
        'unique_models': unique_models, 'per_lambda_freq': per_lambda_freq,
        'bma_pips': bma_pips, 'susie_pips': susie_pips,
        'susie_cs_list': susie_cs_list, 'history': history,
    }


# ---------------------------------------------------------------------------
# Locus fine-mapping orchestrator
# ---------------------------------------------------------------------------

def find_locus_variants(reader, annotation_chr, locus_start, locus_end):
    """
    Find variant indices in the cugen that fall within the locus region.

    Args:
        reader: CugenReader instance
        annotation_chr: DataFrame with annotation for this chromosome
            (must have 'gidx' and 'POS' columns)
        locus_start: start bp of locus
        locus_end: end bp of locus

    Returns:
        local_indices: numpy array of cugen-local variant indices
        annotation_subset: DataFrame of matching annotations
    """
    # Build gidx → POS lookup from annotation
    gidx_to_pos = dict(zip(annotation_chr['gidx'].values,
                           annotation_chr['POS'].values))

    # Get all gidx from cugen
    cugen_gidx = reader.gidx  # numpy int64 array

    # Find which cugen variants are in our locus
    local_indices = []
    matching_gidx = []

    for local_idx in range(len(cugen_gidx)):
        gidx = int(cugen_gidx[local_idx])
        pos = gidx_to_pos.get(gidx, None)
        if pos is not None and locus_start <= pos <= locus_end:
            local_indices.append(local_idx)
            matching_gidx.append(gidx)

    local_indices = np.array(local_indices, dtype=np.int64)
    matching_gidx = np.array(matching_gidx, dtype=np.int64)

    # Get annotation rows for matching variants
    annotation_subset = annotation_chr[
        annotation_chr['gidx'].isin(matching_gidx)
    ].copy()
    # Ensure same order as local_indices
    annotation_subset = annotation_subset.set_index('gidx').loc[matching_gidx].reset_index()

    return local_indices, annotation_subset


def finemap_locus(reader, y_resid, annotation_chr, locus, output_dir,
                  n_pairs=50, n_lambda=20, ridge_alpha=1e-4,
                  susie_L=10, max_variants=15000, device=0,
                  low_memory=False, batch_size=8192,
                  enable_convergence=False, max_rounds=3,
                  discrepancy_threshold=0.3, prior_strength=0.5,
                  tier=2, cache_stats=False, load_stats=False,
                  gwas_z_by_gidx=None, adaptive_weight_gamma=1.0,
                  adaptive_weight_clip=(0.2, 5.0),
                  lambda_min_ratio=0.01,
                  adaptive_weight_method='z',
                  gwas_beta_by_gidx=None, gwas_se_by_gidx=None,
                  gwas_maf_by_gidx=None,
                  lasso_cs_level=60,
                  coding_gidx_set=None, coding_prior_bonus=1.0,
                  verbose=True):
    """
    Run fine-mapping for a single locus.

    Tier 1 (UltraSuSiE): SuSiE-RSS only — fast genome-wide scan.
    Tier 2 (UltraMAP): Dual-engine (SuSiE + LASSO CPSS) with consensus pPIPs.

    Args:
        reader: CugenReader for the chromosome
        y_resid: (n_samples,) numpy array — LOCO residual phenotype
        annotation_chr: DataFrame with annotation for this chromosome
        locus: Series or dict with locus_id, CHR, start_bp, end_bp, etc.
        output_dir: base output directory
        n_pairs: CPSS complementary pairs
        n_lambda: lambda grid size
        ridge_alpha: Elastic Net L2 penalty
        susie_L: number of SuSiE effects
        max_variants: skip locus if more variants than this
        device: GPU device
        low_memory: if True, keep X on CPU and stream to GPU for XtX
        batch_size: samples per GPU batch in low-memory mode
        enable_convergence: if True, run cross-engine convergence (Tier 2)
        max_rounds: max convergence iterations
        discrepancy_threshold: PIP discrepancy threshold for convergence
        prior_strength: blending weight (0=ignore other engine, 1=fully trust)
        tier: 1=UltraSuSiE only, 2=UltraMAP (dual-engine)
        cache_stats: if True, save XtX/Xty to disk for Tier 2 reuse
        load_stats: if True, load cached XtX/Xty instead of re-streaming
        gwas_z_by_gidx: dict gidx -> |z| from step4 GWAS sumstats, or None.
                       When provided, enables adaptive LASSO (Zou 2006):
                       per-variant penalty weight w_j = (median|z|/|z_j|)^gamma,
                       clipped to adaptive_weight_clip. Variants with strong
                       GWAS signal get smaller penalty (easier to select);
                       variants with weak signal get heavier penalty. Breaks
                       LD symmetry and is the correct approach when a
                       consistent initial estimator is available.
        adaptive_weight_gamma: exponent in adaptive penalty (default 1.0)
        adaptive_weight_clip: (min, max) clipping range for weights
        verbose: print progress

    Returns:
        results_df: DataFrame with per-variant results, or None if skipped
    """
    locus_id = int(locus['locus_id'])
    chr_num = int(locus['CHR'])
    start_bp = int(locus['start_bp'])
    end_bp = int(locus['end_bp'])

    if verbose:
        print(f"\n{'='*60}")
        print(f"Locus {locus_id}: chr{chr_num}:{start_bp:,}-{end_bp:,} "
              f"({(end_bp-start_bp)/1000:.0f} kb)")

    # Find variants in locus
    t0 = time.time()
    local_indices, annot_sub = find_locus_variants(
        reader, annotation_chr, start_bp, end_bp)

    n_vars = len(local_indices)
    if verbose:
        print(f"  Found {n_vars} variants in locus")

    if n_vars == 0:
        print(f"  WARNING: No variants found, skipping")
        return None

    if n_vars > max_variants:
        print(f"  WARNING: {n_vars} > {max_variants} max variants, skipping")
        return None

    if n_vars < 2:
        print(f"  WARNING: Only {n_vars} variant(s), skipping")
        return None

    # Get MAF for these variants
    maf = reader.maf[local_indices]
    n_samples = reader.n_samples

    # Center y (common to both paths)
    y_np = y_resid.astype(np.float32)
    y_mean = float(np.mean(y_np))
    y_c_np = y_np - y_mean
    var_y = float(np.var(y_np))

    # Build adaptive-LASSO weights (Zou 2006)
    # Two bases supported:
    #   'z'        : base = |z_j| (classic adaptive LASSO)
    #   'strength' : base = beta_j^2 * 2*af_j*(1-af_j) / se_j^2
    #                (matches step1 top-K selection; does not double-reward
    #                 GWAS significance for rare variants)
    # weight_j = (median(base) / base_j)^gamma, clipped to [w_lo, w_hi].
    adaptive_weights_gpu = None
    n_matched = 0
    method_used = None
    locus_gidx = annot_sub['gidx'].values

    if (adaptive_weight_method == 'strength'
            and gwas_beta_by_gidx is not None
            and gwas_se_by_gidx is not None
            and gwas_maf_by_gidx is not None):
        beta = np.array(
            [gwas_beta_by_gidx.get(int(g), np.nan) for g in locus_gidx],
            dtype=np.float32)
        se = np.array(
            [gwas_se_by_gidx.get(int(g), np.nan) for g in locus_gidx],
            dtype=np.float32)
        mafv = np.array(
            [gwas_maf_by_gidx.get(int(g), np.nan) for g in locus_gidx],
            dtype=np.float32)
        base = (beta ** 2) * (2.0 * mafv * (1.0 - mafv)) / np.maximum(se ** 2, 1e-12)
        n_matched = int(np.sum(~np.isnan(base)))
        if n_matched > 0:
            med_missing = float(np.nanmedian(base))
            base = np.where(np.isnan(base), med_missing, base)
            med_base = float(np.median(base)) if np.median(base) > 0 else 1.0
            weights_np = (med_base / np.maximum(base, 1e-6)) ** adaptive_weight_gamma
            w_lo, w_hi = adaptive_weight_clip
            weights_np = np.clip(weights_np, w_lo, w_hi).astype(np.float32)
            adaptive_weights_gpu = cp.asarray(weights_np)
            method_used = 'strength'
            if verbose:
                print(f"  Adaptive LASSO [strength]: {n_matched}/{n_vars} "
                      f"variants have beta/se/maf; w range=["
                      f"{weights_np.min():.2f}, {weights_np.max():.2f}], "
                      f"median={np.median(weights_np):.2f}")

    if adaptive_weights_gpu is None and gwas_z_by_gidx is not None:
        # Either --adaptive-weight-method z, or 'strength' requested but
        # BETA/SE/MAF inputs unavailable — fall back to |z|.
        if adaptive_weight_method == 'strength':
            if verbose:
                print("  WARNING: --adaptive-weight-method strength requested "
                      "but BETA/SE/MAF not available; falling back to |z|")
        abs_z = np.array(
            [abs(gwas_z_by_gidx.get(int(g), np.nan)) for g in locus_gidx],
            dtype=np.float32)
        n_matched = int(np.sum(~np.isnan(abs_z)))
        if n_matched > 0:
            med_missing = float(np.nanmedian(abs_z))
            abs_z = np.where(np.isnan(abs_z), med_missing, abs_z)
            med_z = float(np.median(abs_z)) if np.median(abs_z) > 0 else 1.0
            weights_np = (med_z / np.maximum(abs_z, 1e-3)) ** adaptive_weight_gamma
            w_lo, w_hi = adaptive_weight_clip
            weights_np = np.clip(weights_np, w_lo, w_hi).astype(np.float32)
            adaptive_weights_gpu = cp.asarray(weights_np)
            method_used = 'z'
            if verbose:
                print(f"  Adaptive LASSO [z]: {n_matched}/{n_vars} variants "
                      f"have GWAS z; w range=[{weights_np.min():.2f}, "
                      f"{weights_np.max():.2f}], median={np.median(weights_np):.2f}")

    # Keep legacy variable name in this scope for the downstream code path
    n_matched_z = n_matched

    # Coding-prior bonus: multiplicatively reduce the adaptive LASSO penalty
    # weight for variants in the coding-gidx set. Orthogonal to z/strength —
    # lets known functional variants (missense/LoF) compete with their
    # LD-tag neighbours even when their data evidence is indistinguishable.
    #   coding_prior_bonus == 1.0  -> no effect (back-compat)
    #   coding_prior_bonus <  1.0  -> lighter penalty for coding variants
    #   coding_prior_bonus >  1.0  -> heavier penalty for coding variants
    # Clips [w_lo, w_hi] re-applied after the multiplier so the adaptive-weight
    # dynamic range cannot blow up.
    if (adaptive_weights_gpu is not None and coding_gidx_set is not None
            and coding_prior_bonus != 1.0):
        is_coding = np.array(
            [1 if int(g) in coding_gidx_set else 0 for g in locus_gidx],
            dtype=bool)
        n_coding = int(is_coding.sum())
        if n_coding > 0:
            w_np = cp.asnumpy(adaptive_weights_gpu).astype(np.float32)
            w_np[is_coding] *= float(coding_prior_bonus)
            w_lo, w_hi = adaptive_weight_clip
            w_np = np.clip(w_np, w_lo, w_hi).astype(np.float32)
            adaptive_weights_gpu = cp.asarray(w_np)
            if verbose:
                print(f"  Coding-prior bonus applied to {n_coding} coding "
                      f"variants (factor {coding_prior_bonus}); w range=["
                      f"{w_np.min():.2f}, {w_np.max():.2f}]")
        elif verbose:
            print("  Coding-prior bonus requested but 0 coding variants "
                  "in locus; no effect.")

    # ------------------------------------------------------------------
    # Compute or load sufficient statistics (XtX, Xty)
    # ------------------------------------------------------------------
    locus_dir = os.path.join(output_dir, f"locus_{locus_id}")
    stats_path = os.path.join(locus_dir, "sufficient_stats.npz")
    X_cpu = None
    X_c = None  # GPU reference for normal-memory Tier 2

    if load_stats and os.path.exists(stats_path):
        # Load cached sufficient statistics
        if verbose:
            print(f"  Loading cached sufficient stats from {stats_path}")
        cached = np.load(stats_path)
        XtX = cp.asarray(cached['XtX'])
        Xty = cp.asarray(cached['Xty'])
        var_y = float(cached['var_y'])
    elif low_memory:
        # ---- LOW-MEMORY PATH: X on CPU, stream to GPU for XtX ----
        if verbose:
            print(f"  [Low-memory mode] Reading genotypes to CPU via GPU chunks...")
        t_load = time.time()
        X_cpu = read_indices_centered_to_cpu(reader, local_indices)
        if verbose:
            print(f"  Loaded {X_cpu.shape} to CPU ({time.time()-t_load:.1f}s, "
                  f"{X_cpu.nbytes/1e9:.1f} GB CPU RAM)")

        t1 = time.time()
        XtX, Xty = compute_split_XtX_batched(
            X_cpu, y_c_np, np.arange(n_samples), batch_size=batch_size)
        if verbose:
            print(f"  Computed XtX ({XtX.shape}) via batched streaming in {time.time()-t1:.1f}s")
    else:
        # ---- NORMAL PATH: X on GPU ----
        X_gpu = reader.read_indices_to_gpu(local_indices)
        if verbose:
            print(f"  Loaded genotypes: {X_gpu.shape} ({time.time()-t0:.1f}s)")

        mu_x = cp.mean(X_gpu, axis=0)
        X_c = X_gpu - mu_x[None, :]
        del X_gpu
        cp.get_default_memory_pool().free_all_blocks()

        y_c = cp.asarray(y_c_np)

        t1 = time.time()
        n = float(n_samples)
        XtX = cp.dot(X_c.T, X_c) / n
        Xty = cp.dot(X_c.T, y_c) / n
        if verbose:
            print(f"  Computed XtX ({XtX.shape}) and Xty in {time.time()-t1:.1f}s")

    # Cache sufficient stats if requested
    if cache_stats:
        os.makedirs(locus_dir, exist_ok=True)
        np.savez_compressed(stats_path,
                            XtX=cp.asnumpy(XtX), Xty=cp.asnumpy(Xty),
                            var_y=var_y, n_samples=n_samples)
        if verbose:
            print(f"  Cached sufficient stats to {stats_path}")

    # ------------------------------------------------------------------
    # Tier 1 (UltraSuSiE): SuSiE-RSS only — fast genome-wide scan
    # ------------------------------------------------------------------
    if tier == 1:
        if verbose:
            print(f"\n  --- Tier 1: UltraSuSiE (SuSiE-RSS only) ---")
        t_b = time.time()
        susie_pips, susie_cs_list = gpu_susie_rss(
            XtX, Xty, var_y, n_samples, L=susie_L, verbose=verbose)
        if verbose:
            print(f"  UltraSuSiE completed in {time.time()-t_b:.1f}s")

        # Zeros for LASSO-side columns (not run in Tier 1)
        sel_freq = np.zeros(n_vars, dtype=np.float64)
        avg_beta = np.zeros(n_vars, dtype=np.float64)
        bma_pips = np.zeros(n_vars, dtype=np.float64)
        per_lambda_freq = np.zeros((1, n_vars), dtype=np.float64)
        lambdas = np.array([0.0])
        joint_r2 = 0.0
        h2_per = np.zeros(n_vars, dtype=np.float64)
        conv_result = None

        # Free genotype memory
        if X_cpu is not None:
            del X_cpu
        if X_c is not None:
            del X_c
        try:
            del y_c
        except NameError:
            pass

    # ------------------------------------------------------------------
    # Tier 2 (UltraMAP): Dual-engine + convergence + consensus pPIPs
    # ------------------------------------------------------------------
    else:
        if enable_convergence:
            if low_memory or load_stats:
                conv_result = cross_engine_convergence(
                    XtX, Xty, var_y, n_samples, n_pairs=n_pairs,
                    n_lambda=n_lambda, ridge_alpha=ridge_alpha, susie_L=susie_L,
                    max_rounds=max_rounds, discrepancy_threshold=discrepancy_threshold,
                    prior_strength=prior_strength,
                    X_cpu=X_cpu, y_cpu=y_c_np, batch_size=batch_size,
                    verbose=verbose)
            else:
                conv_result = cross_engine_convergence(
                    XtX, Xty, var_y, n_samples, n_pairs=n_pairs,
                    n_lambda=n_lambda, ridge_alpha=ridge_alpha, susie_L=susie_L,
                    max_rounds=max_rounds, discrepancy_threshold=discrepancy_threshold,
                    prior_strength=prior_strength,
                    X_gpu=X_c, y_gpu=y_c,
                    verbose=verbose)
        else:
            # Track A: CPSS + BMA
            if verbose:
                print(f"\n  --- Track A: UltraLasso CPSS ---")
            t_a = time.time()
            if low_memory or load_stats:
                if X_cpu is not None:
                    sel_freq, avg_beta, unique_models, per_lambda_freq = \
                        stability_selection_cpss_lowmem(
                            X_cpu, y_c_np, n_pairs=n_pairs, n_lambda=n_lambda,
                            ridge_alpha=ridge_alpha, batch_size=batch_size,
                            weights=adaptive_weights_gpu,
                            lambda_min_ratio=lambda_min_ratio,
                            verbose=verbose)
                else:
                    # load_stats without X — need sufficient-stats-only CPSS
                    # Fall back: use XtX-based stability selection
                    sel_freq, avg_beta, unique_models, per_lambda_freq = \
                        stability_selection_cpss_from_stats(
                            XtX, Xty, var_y, n_samples, n_pairs=n_pairs,
                            n_lambda=n_lambda, ridge_alpha=ridge_alpha,
                            weights=adaptive_weights_gpu,
                            lambda_min_ratio=lambda_min_ratio,
                            verbose=verbose)
            else:
                sel_freq, avg_beta, unique_models, per_lambda_freq = \
                    stability_selection_cpss(
                        X_c, y_c, n_pairs=n_pairs, n_lambda=n_lambda,
                        ridge_alpha=ridge_alpha,
                        weights=adaptive_weights_gpu,
                        lambda_min_ratio=lambda_min_ratio,
                        verbose=verbose)
            bma_pips = compute_bma_pips(
                unique_models, XtX, Xty, var_y, n_samples, verbose=verbose)
            if verbose:
                print(f"  Track A completed in {time.time()-t_a:.1f}s")

            # Track B: GPU SuSiE-RSS
            if verbose:
                print(f"\n  --- Track B: GPU SuSiE ---")
            t_b = time.time()
            susie_pips, susie_cs_list = gpu_susie_rss(
                XtX, Xty, var_y, n_samples, L=susie_L, verbose=verbose)
            if verbose:
                print(f"  Track B completed in {time.time()-t_b:.1f}s")

            conv_result = None

        # R² from sufficient statistics — mid_lambda from the CPSS grid
        lambdas = compute_lambda_grid(Xty, n_lambda=n_lambda,
                                      lambda_min_ratio=lambda_min_ratio,
                                      weights=adaptive_weights_gpu)
        mid_lambda = lambdas[n_lambda // 2]
        theta_lasso, n_act = elastic_fista_sufficient_stats(
            XtX, Xty, n_samples, alpha=mid_lambda, ridge_alpha=ridge_alpha,
            max_iter=500, tol=1e-6)
        if X_c is not None:
            joint_r2, h2_per = compute_variance_explained(X_c, y_c, theta_lasso, maf)
            del X_c, y_c
        else:
            joint_r2, h2_per = compute_variance_explained_from_stats(
                XtX, Xty, var_y, theta_lasso, maf)
            if X_cpu is not None:
                del X_cpu

        # Unpack convergence results if used
        if conv_result is not None:
            sel_freq = conv_result['sel_freq']
            avg_beta = conv_result['avg_beta']
            unique_models = conv_result['unique_models']
            per_lambda_freq = conv_result['per_lambda_freq']
            bma_pips = conv_result['bma_pips']
            susie_pips = conv_result['susie_pips']
            susie_cs_list = conv_result['susie_cs_list']

        if verbose:
            print(f"\n  Locus R² = {joint_r2:.4f} ({n_act} active SNPs at "
                  f"lambda={mid_lambda:.2e})")

    # ------------------------------------------------------------------
    # Build credible sets from sel_freq
    # ------------------------------------------------------------------
    results_df = annot_sub.copy()
    results_df['MAF'] = maf
    results_df['sel_freq'] = sel_freq
    results_df['ultralasso_pip'] = bma_pips
    results_df['susie_pip'] = susie_pips
    results_df['avg_beta'] = avg_beta
    results_df['h2_contribution'] = h2_per

    # Credible sets at various thresholds from sel_freq
    for thresh in [0.50, 0.60, 0.70, 0.80, 0.90]:
        col = f'in_cs_{int(thresh*100)}'
        results_df[col] = (sel_freq >= thresh).astype(int)

    # ------------------------------------------------------------------
    # Consensus Credible Sets (cCS) and pseudo-PIPs (pPIPs)
    # Product formula: pPIP_j = (PIP_susie_j * pi_lasso_j) / sum(...)
    # Concordance = total product mass over cCS
    # Classification: Resolved / Narrowed / Unresolved
    # ------------------------------------------------------------------
    # Build set of local indices that are in any SuSiE credible set
    susie_cs_member_indices = set()
    for cs in susie_cs_list:
        for idx in cs['cs_indices']:
            susie_cs_member_indices.add(idx)

    # in_susie_cs: 1 if variant's local index is in any SuSiE CS
    in_susie_cs = np.array([1 if i in susie_cs_member_indices else 0
                            for i in range(len(annot_sub))], dtype=int)
    results_df['in_susie_cs'] = in_susie_cs

    # Consensus Credible Set: in BOTH LASSO CS (sel_freq >= lasso_cs_level/100)
    # AND SuSiE CS. Lower lasso_cs_level admits more borderline LASSO
    # candidates so the product-formula pPIP spreads rank-order weight across
    # true-ambiguity sites (softmax-like behavior — preserved downstream).
    lasso_cs_col = f'in_cs_{int(lasso_cs_level)}'
    if lasso_cs_col not in results_df.columns:
        raise ValueError(
            f"LASSO CS column {lasso_cs_col} not found (expected "
            f"lasso_cs_level in [50,60,70,80,90])")
    results_df['in_CCS'] = ((results_df[lasso_cs_col] == 1) &
                             (results_df['in_susie_cs'] == 1)).astype(int)

    # Product-formula pPIPs over cCS (memo Section 3.4)
    results_df['consensus_pPIP'] = 0.0
    results_df['concordance'] = 0.0
    results_df['locus_status'] = 'Unresolved'
    ccs_mask = results_df['in_CCS'] == 1
    n_ccs = int(ccs_mask.sum())

    if n_ccs > 0:
        pip_s = results_df.loc[ccs_mask, 'susie_pip'].values
        pi_l = results_df.loc[ccs_mask, 'sel_freq'].values
        products = pip_s * pi_l
        concordance = float(np.sum(products))
        if concordance > 0:
            ppips = products / concordance
        else:
            ppips = np.zeros(n_ccs)
            concordance = 0.0
        results_df.loc[ccs_mask, 'consensus_pPIP'] = ppips
        results_df['concordance'] = concordance

        # Three-outcome classification
        max_ppip = float(np.max(ppips)) if n_ccs > 0 else 0.0
        if n_ccs <= 3 and max_ppip > 0.5 and concordance > 0.3:
            status = 'Resolved'
        elif n_ccs <= 10 and concordance > 0.1:
            status = 'Narrowed'
        else:
            status = 'Unresolved'
        results_df['locus_status'] = status

    # Sort by consensus pPIP first (cCS members on top), then by SuSiE PIP
    results_df = results_df.sort_values(
        ['consensus_pPIP', 'susie_pip'], ascending=[False, False])

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    locus_dir = os.path.join(output_dir, f"locus_{locus_id}")
    os.makedirs(locus_dir, exist_ok=True)

    # Per-variant results
    results_df.to_csv(os.path.join(locus_dir, "variants.tsv"),
                      sep='\t', index=False)

    # Stability data
    np.savez_compressed(
        os.path.join(locus_dir, "stability_data.npz"),
        sel_freq=sel_freq,
        avg_beta=avg_beta,
        per_lambda_freq=per_lambda_freq,
        lambdas=lambdas,
        bma_pips=bma_pips,
        susie_pips=susie_pips,
        h2_per_variant=h2_per,
        joint_r2=joint_r2,
    )

    # SuSiE credible sets
    if susie_cs_list:
        cs_rows = []
        for cs in susie_cs_list:
            for idx in cs['cs_indices']:
                cs_rows.append({
                    'effect_idx': cs['effect_idx'],
                    'variant_local_idx': idx,
                    'gidx': int(annot_sub.iloc[idx]['gidx']),
                    'cs_size': cs['cs_size'],
                    'max_pip': cs['max_pip'],
                })
        if cs_rows:
            pd.DataFrame(cs_rows).to_csv(
                os.path.join(locus_dir, "susie_credible_sets.tsv"),
                sep='\t', index=False)

    # Save convergence history if applicable
    if conv_result is not None and 'history' in conv_result:
        hist_df = pd.DataFrame(conv_result['history'])
        hist_df.to_csv(os.path.join(locus_dir, "convergence_history.tsv"),
                       sep='\t', index=False)

    # Cleanup
    del XtX, Xty
    if tier >= 2:
        del theta_lasso
    gc.collect()
    cp.get_default_memory_pool().free_all_blocks()

    if verbose:
        total_time = time.time() - t0
        n_high_susie = np.sum(susie_pips > 0.5)
        n_cs = len(susie_cs_list)
        cs_sizes = [cs['cs_size'] for cs in susie_cs_list]
        print(f"\n  Locus {locus_id} complete: {total_time:.1f}s")
        print(f"    SuSiE: {n_high_susie} PIP>0.5, {n_cs} credible sets, "
              f"sizes: {cs_sizes}")
        if tier == 2:
            n_high_lasso = np.sum(bma_pips > 0.5)
            n_ccs_out = int(results_df['in_CCS'].sum())
            n_ppip50 = int((results_df['consensus_pPIP'] > 0.5).sum())
            max_ppip = float(results_df['consensus_pPIP'].max())
            concordance_val = float(results_df['concordance'].iloc[0]) if len(results_df) > 0 else 0.0
            status_val = results_df['locus_status'].iloc[0] if len(results_df) > 0 else 'Unresolved'
            print(f"    LASSO: {n_high_lasso} PIP>0.5, "
                  f"sel_freq>=0.6: {int(np.sum(sel_freq>=0.6))}")
            print(f"    cCS: {n_ccs_out} variants, concordance: {concordance_val:.3f}, "
                  f"status: {status_val}")
            print(f"    pPIP>0.5: {n_ppip50}, max pPIP: {max_ppip:.3f}")

    return results_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _process_chromosome_loci(chr_num, chr_loci, cfg, coding_gidx_set,
                             loco_predictions, y_original, annotation_df,
                             worker_id=None):
    """Process all loci on a single chromosome. Shared by sequential + worker paths.

    Returns: (chr_results, chr_summary) — lists ready to accumulate.
    """
    tag = f"[W{worker_id}] " if worker_id is not None else ""
    chr_t0 = time.time()
    print(f"\n{'#'*60}", flush=True)
    print(f"# {tag}Chromosome {chr_num}: {len(chr_loci)} loci", flush=True)
    print(f"{'#'*60}", flush=True)

    cugen_path = os.path.join(cfg['cugen_dir'], f"chr{chr_num}.cugen")
    if not os.path.exists(cugen_path):
        print(f"  {tag}WARNING: {cugen_path} not found, skipping", flush=True)
        return [], []

    reader = CugenReader(cugen_path, device=cfg['device'])
    print(f"  {tag}Cugen: {reader.n_variants:,} variants, "
          f"{reader.n_samples:,} samples", flush=True)

    annotation_chr = annotation_df[
        annotation_df['CHR'] == str(chr_num)
    ].copy()
    print(f"  {tag}Annotation: {len(annotation_chr):,} variants for chr{chr_num}",
          flush=True)

    # GWAS z-scores for adaptive LASSO (per chromosome).
    gwas_z_by_gidx = None
    gwas_beta_by_gidx = None
    gwas_se_by_gidx = None
    gwas_maf_by_gidx = None
    if cfg['gwas_sumstats_dir'] is not None:
        gwas_path = os.path.join(
            cfg['gwas_sumstats_dir'], f"chr{chr_num}_sumstats.tsv")
        if os.path.exists(gwas_path):
            want_cols = ['gidx', 'Z']
            if cfg['adaptive_weight_method'] == 'strength':
                want_cols += ['BETA', 'SE', 'MAF']
            try:
                gwas_df = pd.read_csv(gwas_path, sep='\t', usecols=want_cols)
            except ValueError as exc:
                print(f"  {tag}WARNING: could not read columns {want_cols} "
                      f"from {gwas_path} ({exc}); loading Z only", flush=True)
                gwas_df = pd.read_csv(gwas_path, sep='\t', usecols=['gidx', 'Z'])
            gidx_int = gwas_df['gidx'].astype(int).values
            gwas_z_by_gidx = dict(zip(gidx_int, gwas_df['Z'].values))
            if 'BETA' in gwas_df.columns:
                gwas_beta_by_gidx = dict(zip(gidx_int, gwas_df['BETA'].values))
            if 'SE' in gwas_df.columns:
                gwas_se_by_gidx = dict(zip(gidx_int, gwas_df['SE'].values))
            if 'MAF' in gwas_df.columns:
                gwas_maf_by_gidx = dict(zip(gidx_int, gwas_df['MAF'].values))
            print(f"  {tag}GWAS z-scores: {len(gwas_z_by_gidx):,} from "
                  f"{gwas_path}", flush=True)
        else:
            print(f"  {tag}WARNING: {gwas_path} not found, running without "
                  f"adaptive weights for chr{chr_num}", flush=True)

    y_resid = y_original - loco_predictions[:, chr_num - 1]

    chr_results = []
    chr_summary = []
    for _, locus in chr_loci.iterrows():
        result = finemap_locus(
            reader=reader,
            y_resid=y_resid,
            annotation_chr=annotation_chr,
            locus=locus,
            output_dir=cfg['output_dir'],
            n_pairs=cfg['n_pairs'],
            n_lambda=cfg['n_lambda'],
            ridge_alpha=cfg['ridge_alpha'],
            susie_L=cfg['susie_L'],
            max_variants=cfg['max_variants'],
            device=cfg['device'],
            low_memory=cfg['low_memory'],
            batch_size=cfg['batch_size'],
            enable_convergence=cfg['enable_convergence'],
            max_rounds=cfg['max_rounds'],
            discrepancy_threshold=cfg['discrepancy_threshold'],
            prior_strength=cfg['prior_strength'],
            tier=cfg['tier'],
            cache_stats=cfg['cache_stats'],
            load_stats=cfg['load_stats'],
            gwas_z_by_gidx=gwas_z_by_gidx,
            adaptive_weight_gamma=cfg['adaptive_weight_gamma'],
            adaptive_weight_clip=(cfg['adaptive_weight_clip_min'],
                                  cfg['adaptive_weight_clip_max']),
            lambda_min_ratio=cfg['lambda_min_ratio'],
            adaptive_weight_method=cfg['adaptive_weight_method'],
            gwas_beta_by_gidx=gwas_beta_by_gidx,
            gwas_se_by_gidx=gwas_se_by_gidx,
            gwas_maf_by_gidx=gwas_maf_by_gidx,
            lasso_cs_level=cfg['lasso_cs_level'],
            coding_gidx_set=coding_gidx_set,
            coding_prior_bonus=cfg['coding_prior_bonus'],
            verbose=True,
        )

        if result is not None:
            chr_results.append(result)

            summary_row = {
                'locus_id': int(locus['locus_id']),
                'CHR': int(locus['CHR']),
                'start_bp': int(locus['start_bp']),
                'end_bp': int(locus['end_bp']),
                'n_variants': len(result),
                'n_pip50_susie': int((result['susie_pip'] > 0.5).sum()),
                'max_susie_pip': float(result['susie_pip'].max()),
                'n_susie_cs': int(result['in_susie_cs'].sum()),
            }
            if cfg['tier'] == 2:
                summary_row.update({
                    'n_pip50_lasso': int((result['ultralasso_pip'] > 0.5).sum()),
                    'n_sel_freq60': int((result['sel_freq'] >= 0.6).sum()),
                    'max_lasso_pip': float(result['ultralasso_pip'].max()),
                    'n_in_cCS': int(result['in_CCS'].sum()),
                    'n_pPIP50': int((result['consensus_pPIP'] > 0.5).sum()),
                    'max_pPIP': float(result['consensus_pPIP'].max()),
                    'concordance': float(result['concordance'].iloc[0]) if len(result) > 0 else 0.0,
                    'locus_status': result['locus_status'].iloc[0] if len(result) > 0 else 'Unresolved',
                    'total_h2': float(result['h2_contribution'].sum()),
                })
            chr_summary.append(summary_row)

    reader.close()
    print(f"  {tag}chr{chr_num} done: {len(chr_results)} loci in "
          f"{time.time()-chr_t0:.1f}s", flush=True)
    return chr_results, chr_summary


def _worker_finemap(payload):
    """Worker entry point for multi-process fine-mapping (spawn-safe).

    Receives a chromosome assignment and runs sequentially over its loci.
    Reads all shared state from disk per worker — spawn-safe, no pickle of
    GPU arrays or CuPy handles.
    """
    (worker_id, chrs, cfg, coding_gidx_set_serialized, loco_pred_path,
     annotation_path) = payload

    # Fresh CUDA context per spawned process.
    import cupy as _cp
    _cp.cuda.Device(cfg['device']).use()

    # Load shared read-only state (each worker loads its own copy; ~200 MB).
    loco_data = np.load(loco_pred_path, allow_pickle=True)
    loco_predictions = loco_data['predictions']
    y_original = loco_data['y_original']
    annotation_df = pd.read_feather(annotation_path)
    annotation_df['CHR'] = annotation_df['CHR'].astype(str)

    # Reconstruct coding_gidx_set from serialized form.
    if coding_gidx_set_serialized is None:
        coding_gidx_set = None
    else:
        coding_gidx_set = set(int(g) for g in coding_gidx_set_serialized)

    # loci_df was sliced in main() and passed per-chr as dataframes inside cfg.
    loci_by_chr = cfg.pop('_loci_by_chr')

    worker_results = []
    worker_summary = []
    t_worker_start = time.time()
    for chr_num in chrs:
        chr_loci = loci_by_chr[chr_num]
        res, summ = _process_chromosome_loci(
            chr_num, chr_loci, cfg, coding_gidx_set,
            loco_predictions, y_original, annotation_df,
            worker_id=worker_id)
        worker_results.extend(res)
        worker_summary.extend(summ)
    print(f"  [W{worker_id}] Done: {len(chrs)} chr, "
          f"{len(worker_results)} loci, {time.time()-t_worker_start:.1f}s",
          flush=True)
    return worker_results, worker_summary


def _bin_pack_chromosomes(chr_locus_counts, n_workers):
    """Greedy bin-packing: assign heaviest-unassigned chr to least-loaded worker.
    Load metric = number of loci on that chromosome (proxy for runtime).
    """
    chrs_sorted = sorted(chr_locus_counts.keys(),
                         key=lambda c: chr_locus_counts[c], reverse=True)
    worker_loads = [0] * n_workers
    worker_chrs = [[] for _ in range(n_workers)]
    for c in chrs_sorted:
        idx = worker_loads.index(min(worker_loads))
        worker_chrs[idx].append(c)
        worker_loads[idx] += chr_locus_counts[c]
    return worker_chrs, worker_loads


def main():
    parser = argparse.ArgumentParser(
        description="Step 5b: GPU fine-mapping — UltraSuSiE (Tier 1) + UltraMAP (Tier 2)")
    parser.add_argument("--loci", required=True,
                        help="Path to loci_definition.tsv from step5a")
    parser.add_argument("--loco-pred", required=True,
                        help="Path to loco_predictions.npz from step3")
    parser.add_argument("--cugen-dir", required=True,
                        help="Directory with chr*.cugen files")
    parser.add_argument("--annotation", required=True,
                        help="Path to gidx_annotation.feather")
    parser.add_argument("--output-dir", required=True,
                        help="Output directory for fine-mapping results")
    # Tier selection
    parser.add_argument("--tier", type=int, default=2, choices=[1, 2],
                        help="1 = UltraSuSiE only (SuSiE genome-wide), "
                             "2 = UltraMAP (dual-engine + consensus pPIPs). Default: 2")
    # Sufficient statistics caching
    parser.add_argument("--cache-stats", action="store_true",
                        help="Save per-locus sufficient stats (XtX, Xty) for Tier 2 reuse")
    parser.add_argument("--load-stats", action="store_true",
                        help="Load cached sufficient stats instead of re-streaming genotypes")
    parser.add_argument("--chr", type=int, default=None,
                        help="Process single chromosome (optional)")
    parser.add_argument("--locus-id", type=int, default=None,
                        help="Process single locus by ID (optional)")
    parser.add_argument("--n-pairs", type=int, default=50,
                        help="Number of CPSS complementary pairs (default: 50)")
    parser.add_argument("--n-lambda", type=int, default=20,
                        help="Number of lambda grid points (default: 20)")
    parser.add_argument("--ridge-alpha", type=float, default=1e-4,
                        help="Elastic Net L2 penalty (default: 1e-4)")
    parser.add_argument("--susie-L", type=int, default=10,
                        help="Number of SuSiE effects (default: 10)")
    parser.add_argument("--max-variants", type=int, default=15000,
                        help="Max variants per locus before skipping (default: 15000)")
    parser.add_argument("--device", type=int, default=0,
                        help="GPU device (default: 0)")
    # Low-memory mode
    parser.add_argument("--low-memory", action="store_true",
                        help="Keep X on CPU, stream to GPU for XtX (runs on smaller GPUs)")
    parser.add_argument("--batch-size", type=int, default=8192,
                        help="Samples per GPU batch in low-memory mode (default: 8192)")
    # Cross-engine convergence
    parser.add_argument("--enable-convergence", action="store_true",
                        help="Enable iterative cross-engine convergence")
    parser.add_argument("--max-rounds", type=int, default=3,
                        help="Max convergence rounds (default: 3)")
    parser.add_argument("--discrepancy-threshold", type=float, default=0.3,
                        help="PIP discrepancy threshold to trigger convergence (default: 0.3)")
    parser.add_argument("--prior-strength", type=float, default=0.5,
                        help="Blending weight for cross-engine priors 0-1 (default: 0.5)")
    # Adaptive LASSO (Zou 2006) using GWAS z-scores
    parser.add_argument("--gwas-sumstats-dir", default=None,
                        help="Directory containing chr{N}_sumstats.tsv with "
                             "CHR/POS/ID/BETA/SE/Z/gidx columns from step4. "
                             "When provided, CPSS LASSO uses adaptive weights "
                             "w_j = (median|z| / |z_j|)^gamma.")
    parser.add_argument("--adaptive-weight-gamma", type=float, default=1.0,
                        help="Exponent in adaptive penalty weight (default 1.0)")
    parser.add_argument("--adaptive-weight-clip-min", type=float, default=0.2,
                        help="Floor on adaptive weights (default 0.2)")
    parser.add_argument("--adaptive-weight-clip-max", type=float, default=5.0,
                        help="Ceiling on adaptive weights (default 5.0)")
    parser.add_argument("--lambda-min-ratio", type=float, default=0.01,
                        help="lambda_min = lambda_max * ratio for CPSS grid "
                             "(default 0.01). Raise to 0.1-0.3 in dense-LD "
                             "high-variable loci where LASSO saturates at the "
                             "low-lambda end and unique_models stays at 0.")
    parser.add_argument("--adaptive-weight-method", choices=['z', 'strength'],
                        default='z',
                        help="Adaptive LASSO penalty basis. 'z' uses |z| "
                             "(back-compat default). 'strength' uses "
                             "beta^2 * 2*af*(1-af) / se^2 (matches step1 "
                             "top-K criterion, avoids double-rewarding GWAS "
                             "significance for rare variants). Requires "
                             "BETA/SE/MAF columns in sumstats; falls back to "
                             "'z' if any are missing.")
    parser.add_argument("--lasso-cs-level", type=int, default=60,
                        choices=[50, 60, 70, 80, 90],
                        help="LASSO sel_freq threshold (%%) for first-pass "
                             "inclusion into consensus credible set (cCS = "
                             "LASSO CS INTERSECT SuSiE CS). Default 60. Lower "
                             "values (e.g. 50) admit more borderline LASSO "
                             "candidates so the product-formula pPIP can "
                             "spread mass across true-ambiguity sites as a "
                             "soft rank-order weight.")
    parser.add_argument("--coding-gidx", default=None,
                        help="Path to coding_gidx_*.npz produced by "
                             "prebuild_coding_annot.py. Contains 'gidx' and "
                             "'is_high_impact' arrays.")
    parser.add_argument("--coding-prior-bonus", type=float, default=1.0,
                        help="Multiplicative factor applied to the adaptive "
                             "LASSO weight w_j of coding (HIGH/MODERATE "
                             "impact) variants. Default 1.0 = no effect. "
                             "Use <1.0 (e.g. 0.25) to favour coding variants "
                             "over their LD-tag neighbours; >1.0 to penalise "
                             "them. Clips re-applied after the multiplier.")
    # Multi-worker parallelism (session 14).
    parser.add_argument("--n-workers", type=int, default=1,
                        help="Number of parallel GPU worker processes (CUDA MPS). "
                             "Default 1 = sequential (back-compat). "
                             "Recommended 8 on H100 SXM5 with --cpus-per-task>=16 "
                             "and MPS started in the SLURM wrapper. Workers are "
                             "bin-packed by locus count per chromosome; each "
                             "worker opens its own CugenReader.")
    args = parser.parse_args()

    cp.cuda.Device(args.device).use()
    os.makedirs(args.output_dir, exist_ok=True)

    tier_name = "UltraSuSiE" if args.tier == 1 else "UltraMAP"
    print(f"\n{'='*60}")
    print(f"Step 5b: {tier_name} (Tier {args.tier}) GPU Fine-Mapping")
    print(f"{'='*60}")
    if args.cache_stats:
        print("  [Cache mode: will save sufficient stats per locus]")
    if args.load_stats:
        print("  [Load mode: will load cached sufficient stats]")

    # Optional coding-prior lookup — loaded once, shared across all loci
    coding_gidx_set = None
    if args.coding_gidx is not None and args.coding_prior_bonus != 1.0:
        if not os.path.exists(args.coding_gidx):
            raise FileNotFoundError(
                f"--coding-gidx file not found: {args.coding_gidx}")
        cg = np.load(args.coding_gidx)
        # Use HIGH/MODERATE-impact subset if available, else all coding gidx
        if 'is_high_impact' in cg.files:
            mask = cg['is_high_impact']
            coding_gidx_set = set(int(x) for x in cg['gidx'][mask])
            print(f"  Coding-prior bonus {args.coding_prior_bonus}x applied "
                  f"to {len(coding_gidx_set):,} HIGH-impact variants from "
                  f"{args.coding_gidx}")
        else:
            coding_gidx_set = set(int(x) for x in cg['gidx'])
            print(f"  Coding-prior bonus {args.coding_prior_bonus}x applied "
                  f"to {len(coding_gidx_set):,} coding variants from "
                  f"{args.coding_gidx}")
    elif args.coding_prior_bonus != 1.0:
        print(f"  WARNING: --coding-prior-bonus {args.coding_prior_bonus} "
              f"requires --coding-gidx; bonus ignored.")

    # ---- Load inputs ----
    print("\nLoading loci definition...")
    loci_df = pd.read_csv(args.loci, sep='\t')
    print(f"  {len(loci_df)} loci loaded")

    # Filter by chromosome if specified
    if args.chr is not None:
        loci_df = loci_df[loci_df['CHR'] == args.chr].copy()
        print(f"  Filtered to chr{args.chr}: {len(loci_df)} loci")

    # Filter by locus ID if specified
    if args.locus_id is not None:
        loci_df = loci_df[loci_df['locus_id'] == args.locus_id].copy()
        print(f"  Filtered to locus {args.locus_id}: {len(loci_df)} loci")

    if len(loci_df) == 0:
        print("No loci to process. Exiting.")
        return

    print("\nLoading LOCO predictions...")
    loco_data = np.load(args.loco_pred, allow_pickle=True)
    loco_predictions = loco_data['predictions']  # (n_samples, 22)
    y_original = loco_data['y_original']         # (n_samples,)
    fid_order = loco_data['fid_order']           # (n_samples,)
    print(f"  {len(y_original):,} samples, {loco_predictions.shape[1]} chromosomes")

    print("\nLoading annotation...")
    annotation_df = pd.read_feather(args.annotation)
    annotation_df['CHR'] = annotation_df['CHR'].astype(str)
    print(f"  {len(annotation_df):,} annotated variants")

    # ---- Build shared config dict (flags) for sequential + worker paths ----
    chrs_sorted = sorted(loci_df['CHR'].unique())
    loci_by_chr = {int(c): loci_df[loci_df['CHR'] == c].copy() for c in chrs_sorted}
    cfg = {
        'cugen_dir': args.cugen_dir,
        'output_dir': args.output_dir,
        'gwas_sumstats_dir': args.gwas_sumstats_dir,
        'n_pairs': args.n_pairs,
        'n_lambda': args.n_lambda,
        'ridge_alpha': args.ridge_alpha,
        'susie_L': args.susie_L,
        'max_variants': args.max_variants,
        'device': args.device,
        'low_memory': args.low_memory,
        'batch_size': args.batch_size,
        'enable_convergence': args.enable_convergence,
        'max_rounds': args.max_rounds,
        'discrepancy_threshold': args.discrepancy_threshold,
        'prior_strength': args.prior_strength,
        'tier': args.tier,
        'cache_stats': args.cache_stats,
        'load_stats': args.load_stats,
        'adaptive_weight_gamma': args.adaptive_weight_gamma,
        'adaptive_weight_clip_min': args.adaptive_weight_clip_min,
        'adaptive_weight_clip_max': args.adaptive_weight_clip_max,
        'lambda_min_ratio': args.lambda_min_ratio,
        'adaptive_weight_method': args.adaptive_weight_method,
        'lasso_cs_level': args.lasso_cs_level,
        'coding_prior_bonus': args.coding_prior_bonus,
    }

    all_results = []
    loci_summary = []
    total_t0 = time.time()

    if args.n_workers <= 1:
        # ---- Sequential path (back-compat) ----
        for chr_num in chrs_sorted:
            chr_num = int(chr_num)
            res, summ = _process_chromosome_loci(
                chr_num, loci_by_chr[chr_num], cfg, coding_gidx_set,
                loco_predictions, y_original, annotation_df)
            all_results.extend(res)
            loci_summary.extend(summ)
    else:
        # ---- Multi-worker path: bin-pack chromosomes by locus count ----
        chr_locus_counts = {int(c): len(loci_by_chr[int(c)]) for c in chrs_sorted}
        n_workers = min(args.n_workers, len(chrs_sorted))
        worker_chrs, worker_loads = _bin_pack_chromosomes(
            chr_locus_counts, n_workers)
        print(f"\nMulti-worker dispatch: {n_workers} workers "
              f"(requested {args.n_workers})")
        for wi, (ws, ld) in enumerate(zip(worker_chrs, worker_loads)):
            print(f"  [W{wi}] chr{ws} — {ld} loci")

        # Serialize coding_gidx_set for pickling to workers (sets ARE picklable
        # but we keep consistent with the worker's reconstruct-from-list path).
        coding_gidx_list = (None if coding_gidx_set is None
                            else list(coding_gidx_set))

        # Per-worker payload: each worker gets its own chr assignment + a copy
        # of cfg with the chr-sliced loci subset for those chromosomes only.
        payloads = []
        for wi, chrs_for_w in enumerate(worker_chrs):
            if not chrs_for_w:
                continue
            cfg_w = dict(cfg)
            cfg_w['_loci_by_chr'] = {c: loci_by_chr[c] for c in chrs_for_w}
            payloads.append((
                wi, chrs_for_w, cfg_w, coding_gidx_list,
                args.loco_pred, args.annotation,
            ))

        ctx = mp.get_context('spawn')
        with ctx.Pool(processes=len(payloads)) as pool:
            for (worker_results, worker_summary) in pool.imap_unordered(
                    _worker_finemap, payloads):
                all_results.extend(worker_results)
                loci_summary.extend(worker_summary)

    # ---- Save combined outputs ----
    total_time = time.time() - total_t0
    print(f"\n{'='*60}")
    print(f"Fine-mapping complete: {len(all_results)} loci in {total_time:.1f}s")

    if loci_summary:
        summary_df = pd.DataFrame(loci_summary)
        summary_path = os.path.join(args.output_dir, "loci_summary.tsv")
        summary_df.to_csv(summary_path, sep='\t', index=False)
        print(f"Saved loci summary to {summary_path}")

        # Print summary table
        if args.tier == 1:
            print(f"\n{'Locus':>6} {'CHR':>3} {'Vars':>5} {'PIP>.5':>7} "
                  f"{'MaxPIP':>7} {'nCS':>4}")
            print("-" * 40)
            for _, row in summary_df.iterrows():
                print(f"{int(row['locus_id']):>6d} {int(row['CHR']):>3d} "
                      f"{int(row['n_variants']):>5d} "
                      f"{int(row['n_pip50_susie']):>7d} "
                      f"{row['max_susie_pip']:>7.3f} "
                      f"{int(row['n_susie_cs']):>4d}")
        else:
            print(f"\n{'Locus':>6} {'CHR':>3} {'Vars':>5} {'PIP(S)':>7} "
                  f"{'PIP(L)':>7} {'cCS':>4} {'Conc':>6} {'Status':>11}")
            print("-" * 65)
            for _, row in summary_df.iterrows():
                print(f"{int(row['locus_id']):>6d} {int(row['CHR']):>3d} "
                      f"{int(row['n_variants']):>5d} "
                      f"{int(row['n_pip50_susie']):>7d} "
                      f"{int(row['n_pip50_lasso']):>7d} "
                      f"{int(row['n_in_cCS']):>4d} "
                      f"{row['concordance']:>6.3f} "
                      f"{row['locus_status']:>11s}")

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined_path = os.path.join(args.output_dir, "all_variants.tsv.gz")
        combined.to_csv(combined_path, sep='\t', index=False, compression='gzip')
        print(f"Saved {len(combined):,} variants to {combined_path}")

    print(f"\nTotal wall time: {total_time:.1f}s")


if __name__ == "__main__":
    main()
