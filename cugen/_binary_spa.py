"""cugen._binary_spa — GPU logistic score test + selective saddlepoint (SPA).

Self-contained, vendored copy of the binary-trait score/SPA kernels from the
production ``ultralasso_binary_utils.py`` (the validated SAIGE-style binary GWAS
used for the manuscript). No ``sys.path`` injection, no import from repo-root
scripts — this module is fully owned by the package (per the self-contained rule).

Only the kernels needed by Step-4 binary association are vendored:
  * ``sigmoid_gpu``                    — numerically stable logistic link
  * ``_batched_Xv``                    — batched X @ v (avoids CuPy f64 upcast OOM)
  * ``compute_binary_score_stats_gpu`` — logistic score U, adjusted variance V,
                                         beta, SE, Z, normal-tail p, mu_g
  * ``spa_pvalues_centered_gpu``       — Lugannani–Rice saddlepoint p-values
                                         (sign-preserving denominator — keeps the
                                         negative-beta fix from the SPA sign bug)
  * ``apply_fast_spa_gpu``             — selective SPA over a chunked mask

The null model is supplied entirely through ``eta`` (logit scale): for the
LPM-residual production path, ``eta = logit(clip(mu_LPM))`` so ``sigmoid(eta)``
recovers the LPM null probability and the logistic score collapses to the
LPM-residual score U = X'(y - mu). See ``assoc._worker_gwas_binary``.

CuPy is imported at module top — therefore this module must only ever be
imported INSIDE a spawned worker (never at the parent-process module top of
``assoc``), so the parent keeps a clean CUDA context for ``spawn``.
"""
from __future__ import annotations

import math
from typing import Dict

import cupy as cp
from cupyx.scipy.special import erf as _gpu_erf, erfc as _gpu_erfc


EPS32 = 1e-12
EPS64 = 1e-18
SQRT2 = math.sqrt(2.0)


def sigmoid_gpu(x: cp.ndarray) -> cp.ndarray:
    """Numerically stable sigmoid on GPU."""
    return 0.5 * (cp.tanh(x * 0.5) + 1.0)


def _batched_Xv(X_gpu: cp.ndarray, v_gpu: cp.ndarray, batch_size: int = 4096) -> cp.ndarray:
    """Compute X @ v in column batches to avoid CuPy float64 upcasting OOM."""
    n = X_gpu.shape[0]
    p = X_gpu.shape[1]
    out = cp.zeros(n, dtype=cp.float32)
    v_gpu = v_gpu.astype(cp.float32, copy=False)
    for s in range(0, p, batch_size):
        e = min(s + batch_size, p)
        out += X_gpu[:, s:e] @ v_gpu[s:e]
    return out


def compute_binary_score_stats_gpu(
    X_gpu: cp.ndarray,
    y_gpu: cp.ndarray,
    eta_gpu: cp.ndarray,
) -> Dict[str, cp.ndarray]:
    """
    Compute logistic score statistics for one block of variants on GPU.

    The null model is specified entirely by eta_gpu (which already contains
    covariates + LOCO offset). We only re-center genotypes for the intercept
    nuisance term.
    """
    mu = sigmoid_gpu(eta_gpu.astype(cp.float32, copy=False))
    w = cp.maximum(mu * (1.0 - mu), EPS32)
    r = y_gpu.astype(cp.float32, copy=False) - mu

    # Accumulate the three score reductions (U_raw=X'r, wg=X'w, wxx=(X^2)'w) in
    # FLOAT64, batched over columns so no dense float64 X (or X*X) is ever
    # materialised — the single-process legacy uses one big float32 cuBLAS call,
    # but the SOTA 8-worker step4 (8 CUDA-MPS workers on one 80 GB GPU) cannot
    # afford it. Two reasons for float64, not just memory:
    #  (1) the variance V = wxx - wg^2/sum_w is a CATASTROPHIC CANCELLATION of two
    #      large near-equal sums; in float32 it differs by up to ~5% on low-
    #      variance variants and is reduction-ORDER-dependent (8-worker batched
    #      vs single-process full give different roundings).
    #  (2) the SOTA continuous fused_univariate kernel already accumulates in
    #      `double` for exactly this reason — we match that precision discipline.
    # float64 makes V accurate AND layout-independent, so the 8-worker result is
    # reproducible and does not depend on the column batching.
    w64 = w.astype(cp.float64)
    r64 = r.astype(cp.float64)
    sum_w = cp.sum(w64)
    sum_r = cp.sum(r64)

    m = X_gpu.shape[1]
    U_raw = cp.empty(m, dtype=cp.float64)
    wg = cp.empty(m, dtype=cp.float64)
    wxx = cp.empty(m, dtype=cp.float64)
    _WB = 256
    for _s in range(0, m, _WB):
        _e = min(_s + _WB, m)
        _Xs = X_gpu[:, _s:_e].astype(cp.float64)
        U_raw[_s:_e] = _Xs.T @ r64
        wg[_s:_e] = _Xs.T @ w64
        wxx[_s:_e] = (_Xs * _Xs).T @ w64
        del _Xs

    sw = cp.maximum(sum_w, EPS64)
    mu_g = wg / sw
    U = U_raw - mu_g * sum_r
    V = cp.maximum(wxx - (wg * wg) / sw, EPS64)
    beta = U / V
    se = 1.0 / cp.sqrt(V)
    z = U / cp.sqrt(V)
    p_norm = _gpu_erfc(cp.abs(z) / SQRT2)

    return {
        "mu": mu,
        "w": w,
        "r": r,
        "sum_w": sum_w,
        "sum_r": sum_r,
        "U": U,
        "V": V,
        "beta": beta,
        "se": se,
        "z": z,
        "p_norm": p_norm,
        "mu_g": mu_g,
    }


def spa_pvalues_centered_gpu(
    G_centered_gpu: cp.ndarray,
    mu_gpu: cp.ndarray,
    U_gpu: cp.ndarray,
    max_iter: int = 50,
    tol: float = 1e-8,
) -> cp.ndarray:
    """
    Vectorized saddlepoint approximation for a block of centered genotype columns.

    Args:
        G_centered_gpu: centered genotypes, shape (n_samples, m)
        mu_gpu: null probabilities, shape (n_samples,)
        U_gpu: observed score statistics for each column, shape (m,)

    Returns:
        Two-sided SPA p-values on GPU.
    """
    if G_centered_gpu.size == 0:
        return cp.zeros(0, dtype=cp.float64)

    G = G_centered_gpu.astype(cp.float64, copy=False)
    mu = mu_gpu.astype(cp.float64, copy=False)[:, None]
    U = U_gpu.astype(cp.float64, copy=False)

    V = cp.sum(mu * (1.0 - mu) * G * G, axis=0)
    z = U / cp.sqrt(cp.maximum(V, EPS64))
    p_norm = _gpu_erfc(cp.abs(z) / SQRT2)

    muG = cp.sum(mu * G, axis=0)
    t = cp.clip(U / cp.maximum(V, EPS64), -1.0, 1.0)

    for _ in range(max_iter):
        tg = cp.clip(G * t[None, :], -60.0, 60.0)
        exp_tg = cp.exp(tg)
        denom = 1.0 - mu + mu * exp_tg
        q = (mu * exp_tg) / cp.maximum(denom, EPS64)

        Kp = cp.sum(G * (q - mu), axis=0)
        Kpp = cp.sum(G * G * q * (1.0 - q), axis=0)
        step = (Kp - U) / cp.maximum(Kpp, EPS64)
        t_new = cp.clip(t - step, -20.0, 20.0)

        if float(cp.max(cp.abs(t_new - t))) < tol:
            t = t_new
            break
        t = t_new

    tg = cp.clip(G * t[None, :], -60.0, 60.0)
    exp_tg = cp.exp(tg)
    denom = 1.0 - mu + mu * exp_tg
    q = (mu * exp_tg) / cp.maximum(denom, EPS64)
    K = cp.sum(cp.log1p(mu * (exp_tg - 1.0)), axis=0) - t * muG
    Kpp = cp.sum(G * G * q * (1.0 - q), axis=0)

    arg = 2.0 * (t * U - K)
    valid = (arg > 0.0) & (Kpp > 0.0) & (cp.abs(t) > 1e-10)
    w = cp.sign(t) * cp.sqrt(cp.maximum(arg, EPS64))
    v = t * cp.sqrt(cp.maximum(Kpp, EPS64))
    safe_w = cp.where(cp.abs(w) > 1e-12, w, 1.0)  # |w|<1e-12 masked by `valid`
    r = w + cp.log(cp.maximum(cp.abs(v / safe_w), EPS64)) / safe_w
    cdf_r = 0.5 * (1.0 + _gpu_erf(r / SQRT2))
    p_one = cp.where(U >= 0.0, 1.0 - cdf_r, cdf_r)
    p_two = 2.0 * cp.minimum(p_one, 1.0 - p_one)
    p_two = cp.clip(p_two, 1e-300, 1.0)

    return cp.where(valid, p_two, p_norm.astype(cp.float64))


def apply_fast_spa_gpu(
    X_gpu: cp.ndarray,
    score_stats: Dict[str, cp.ndarray],
    spa_mask: cp.ndarray,
    spa_chunk: int = 64,
) -> cp.ndarray:
    """
    Apply SPA only to flagged variants in the current block.
    Returns p-values for all variants (normal approx elsewhere).
    """
    p_final = score_stats["p_norm"].astype(cp.float64)
    if int(cp.sum(spa_mask)) == 0:
        return p_final

    mu = score_stats["mu"]
    mu_g = score_stats["mu_g"]
    U = score_stats["U"]
    flag_idx = cp.where(spa_mask)[0]

    for start in range(0, int(flag_idx.shape[0]), spa_chunk):
        idx = flag_idx[start:start + spa_chunk]
        G = X_gpu[:, idx].astype(cp.float64)
        G -= mu_g[idx][None, :].astype(cp.float64)
        p_chunk = spa_pvalues_centered_gpu(G, mu, U[idx].astype(cp.float64))
        p_final[idx] = p_chunk
        del G, p_chunk
        cp.get_default_memory_pool().free_all_blocks()

    return p_final
