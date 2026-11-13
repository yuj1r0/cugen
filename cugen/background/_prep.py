"""Covariates, phenotype/genotype standardisation, folds, LOO hat diagonal.

REGENIE conventions (v2.2.4) are the defaults:
  * phenotypes  y - Q Q'y, scaled by ||.|| / sqrt(n - C)            (Pheno.cpp 1270-1284)
  * genotypes   mean-impute missing over the analysed samples, G - (G Q) Q', scaled by
                ||.|| / sqrt(n - C)                                   (Data.cpp 159-180)
  * folds       contiguous in sample order, floor(n/K) each, last fold takes the rest
                                                                      (Data.cpp 395-420)
``scaling="gcta"`` standardises a genotype as (g - 2p) / sqrt(2p(1-p)) with missing -> 0 and
no covariate projection (GCTA GRM convention; used by UltraLOCO).
"""
from __future__ import annotations

import warnings

import numpy as np


def as_matrix(y, name="y", check_binary=False):
    """y (n,) / (n, P) / pandas -> float64 (n, P) array and phenotype names."""
    names = None
    if hasattr(y, "columns"):
        names = [str(c) for c in y.columns]
        y = y.to_numpy(np.float64)
    elif hasattr(y, "name") and hasattr(y, "to_numpy"):
        names = [str(y.name) if y.name is not None else "y"]
        y = y.to_numpy(np.float64)
    Y = np.asarray(y, dtype=np.float64)
    if Y.ndim == 1:
        Y = Y[:, None]
    if Y.ndim != 2:
        raise ValueError(f"{name} must be (n,) or (n, P)")
    if names is None:
        names = ["y"] if Y.shape[1] == 1 else [f"y{k}" for k in range(Y.shape[1])]
    if not np.isfinite(Y).all():
        raise ValueError(f"{name} has missing/non-finite values; subset the samples to complete "
                         f"cases first (missing phenotypes are not handled by the background engines)")
    if check_binary:
        warn_if_binary(Y, names)
    return Y, names


def binary_columns(Y):
    """Indices of columns holding exactly two distinct values (0/1, 1/2 case-control codes...)."""
    out = []
    for k in range(Y.shape[1]):
        u = np.unique(Y[:, k])
        if len(u) == 2:
            out.append(k)
    return out


def warn_if_binary(Y, names):
    """A two-valued phenotype gets a LINEAR (linear-probability) background here; warn."""
    b = binary_columns(Y)
    if b:
        warnings.warn(
            f"phenotype(s) {[names[k] for k in b]} take only two values: the background is fitted as a "
            f"linear-probability model (a logistic level 1, as in REGENIE --bt, is not implemented). "
            f"For case-control traits use the result as a linear background only.",
            UserWarning, stacklevel=4)
    return b


def design(covariates, n, intercept=True):
    """[1, covariates] as float64 (n, k)."""
    cols = [np.ones((n, 1))] if intercept else []
    if covariates is not None:
        C = covariates.to_numpy(np.float64) if hasattr(covariates, "to_numpy") else np.asarray(covariates, np.float64)
        if C.ndim == 1:
            C = C[:, None]
        if C.shape[0] != n:
            raise ValueError(f"covariates have {C.shape[0]} rows, phenotype has {n}")
        if not np.isfinite(C).all():
            raise ValueError("covariates have missing/non-finite values")
        cols.append(C)
    if not cols:
        return np.zeros((n, 0))
    return np.column_stack(cols)


def covariate_basis(covariates, n, intercept=True, tol=1e-8):
    """Orthonormal basis Q (n, ncov) of [1, covariates].

    Full rank: plain QR (columns kept where |R_jj| > tol max|R|), which keeps results
    bit-compatible with the validated reference implementation. Rank deficient:
    column-pivoted QR, keeping the numerical rank."""
    X0 = design(covariates, n, intercept)
    if X0.shape[1] == 0:
        return np.zeros((n, 0)), X0
    Q, R = np.linalg.qr(X0)
    d = np.abs(np.diag(R))
    keep = d > tol * np.abs(R).max()
    if keep.all():
        return Q, X0
    import scipy.linalg as sl
    Q, R, _ = sl.qr(X0, mode="economic", pivoting=True)
    rk = int((np.abs(np.diag(R)) > tol * np.abs(R).max()).sum())
    return Q[:, :rk], X0


def residualize_scale(Y, Q):
    """REGENIE residualize_phenotypes: (Ys, scale, yhat_cov) with Y = yhat_cov + scale * Ys."""
    n, ncov = Q.shape
    Yres = Y - Q @ (Q.T @ Y)
    scale = np.sqrt((Yres ** 2).sum(0) / (n - ncov))
    if np.any(scale <= 0):
        raise ValueError("a phenotype is constant after removing covariates")
    return Yres / scale, scale, Y - Yres


def fold_index(n, k=5, kind="contiguous", seed=0):
    """List of K sample-index arrays. 'contiguous' = REGENIE set_folds; 'random' = a seeded
    balanced random assignment (np.random.default_rng(seed).permutation(arange(n) % K))."""
    if kind == "contiguous":
        t = n // k
        sizes = [t] * (k - 1) + [n - t * (k - 1)]
        b = np.cumsum([0] + sizes)
        return [np.arange(b[i], b[i + 1]) for i in range(k)]
    if kind == "random":
        lab = np.random.default_rng(seed).permutation(np.arange(n) % k)
        return [np.flatnonzero(lab == i) for i in range(k)]
    raise ValueError("fold kind must be 'contiguous' or 'random'")


def standardize_regenie(counts, Q, ncov, xp, dtype=None, lowvar=1e-6):
    """counts (b, n) uint8 (3 = missing) -> (G (b, n), ok (b,)).

    Mean-impute over the analysed samples, project out Q, scale to ||g||^2 = n - ncov.
    Low-variance rows (sd < lowvar after projection) or all-missing rows are zeroed and
    flagged ok = False."""
    dtype = dtype or xp.float64
    n = counts.shape[1]
    lut = xp.asarray(np.array([0.0, 1.0, 2.0, 0.0]), dtype=dtype)
    g = lut[counts]
    miss = counts == 3
    nmiss = miss.sum(1)
    cnt = n - nmiss
    mean = g.sum(1) / xp.maximum(cnt, 1)
    if bool((nmiss > 0).any()):
        g = xp.where(miss, mean[:, None].astype(dtype), g)
    if Q.shape[1]:
        g -= (g @ Q) @ Q.T
    sc = xp.sqrt((g * g).sum(1) / (n - ncov))
    ok = (sc > lowvar) & (cnt > 0)
    g = xp.where(ok[:, None], g / xp.where(ok, sc, 1.0)[:, None], 0.0)
    return g, ok


def standardize_gcta(counts, xp, dtype=None):
    """counts (b, n) -> (Z (b, n), valid-mask (b, n) as dtype, ok (b,)), GCTA GRM convention:
    p = mean/2 over valid calls, z = (g - 2p)/sqrt(2p(1-p)), missing -> 0, monomorphic rows
    dropped (ok = False)."""
    dtype = dtype or xp.float64
    miss = counts == 3
    lut = xp.asarray(np.array([0.0, 1.0, 2.0, 0.0]), dtype=dtype)
    gf = lut[counts]
    nval = (~miss).sum(1)
    p = gf.sum(1) / xp.maximum(nval, 1) / 2
    ok = (p > 0) & (p < 1)
    z = xp.where(miss, 0, (gf - 2 * p[:, None]) / xp.sqrt(xp.maximum(2 * p * (1 - p), 1e-12))[:, None])
    return z.astype(dtype, copy=False), (~miss).astype(dtype), ok


def loo_hat_diag(U, F, hQ):
    """Diagonal of the full hat matrix H(lam) = Q Q' + U diag(F) U' for each lambda column of F
    (n, L): covariate leverage hQ_ii plus the kernel-ridge part."""
    return (U * U) @ F + hQ[:, None]
