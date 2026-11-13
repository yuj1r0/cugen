"""CuGenie-K: whole-genome kernel-ridge LOCO in sample space.

One pass over the genotypes accumulates per-chromosome Gram matrices S_c of the
REGENIE-standardised genotypes (mean-imputed, covariates projected out, ||z||^2 = n - ncov).
LOCO for chromosome c is a refit on K_-c = (S - S_c)/M_-c: one f64 eigh, H(lam) =
U diag(d/(d+lam)) U', lambda chosen per chromosome by exact leave-one-out CV (closed form from
the full hat diagonal, covariate leverage included). Offsets: ``offset="loo"`` (default)
y - (y - Hy)/(1 - diag H), or ``offset="insample"`` Hy; ``"both"`` returns both.

Calibration note: with the plain offset step 4, CuGenie-K deflates a structured null
(lambda_GC ~0.92 in our tests); CuGenie (REGENIE-style) and UltraLOCO (x-side GLS whitening
on the same kind of kernel; ``fit_ultraloco(scaling="regenie")`` is the MLMA-whitened
CuGenie-K) are the calibrated small-n choices.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .. import _resources as R
from .._device import device as _device
from ._kernels import KernelSet, kernel_bytes
from ._prep import as_matrix, covariate_basis, residualize_scale
from .loco import build as _build_loco
from .loco import output_chromosomes

logger = logging.getLogger("cugen.background.cugenie_k")

LAM_WIDE = np.logspace(-7, 3, 101)      # 1e-7 .. 1e3, 101 points: reproduces the validated reference
LAM_STABLE = 1e-5                       # below this the closed-form LOO-CV curve is rounding-dominated
LAM_GRIDS = {"wide": LAM_WIDE,                                     # default
             "stable": np.logspace(np.log10(LAM_STABLE), 3, 81)}   # implementation-stable lambda choice


def _log(verbose, *a):
    if verbose:
        logger.info(" ".join(str(x) for x in a))


def _auto_block(n, budget, n_file, itemsize=8):
    per = n * itemsize * 4 + ((n_file + 3) // 4) * 4 * 2
    return int(max(256, min(16384, (budget * 0.15) // max(per, 1))))


def estimate_cugenie_k(n, M, n_chrom=38, P=1, n_lam=101, low_memory=False, gram_dtype="float64",
                       block_size=4096, n_file=None):
    n_file = n_file or n
    isz = 4 if gram_dtype == "float32" else 8
    block_size = max(1, min(block_size, M))
    kern = kernel_bytes(n, n_chrom, False, low_memory, work=6)
    blk = block_size * n * (8 * 3 + isz) + block_size * ((n_file + 3) // 4) * 4 * 2
    cv = 3 * n * n_lam * P * 8 + 2 * n * n_lam * 8
    off = 3 * n * n_chrom * P * 8
    peak = int(kern + max(blk, cv) + off + 64e6)
    fl_pass = 2.0 * M * n * n * (1 if isz == 8 else 0.5) * (2 if low_memory else 1)
    fl_eig = n_chrom * 10.0 * n ** 3
    st = {"kernel_pass": fl_pass, "eigh": fl_eig}
    return {"method": "cugenie_k", "n": n, "M": M, "peak_bytes": peak,
            "peak_breakdown": {"kernels": kern, "block": blk, "cv": cv},
            "flops": float(sum(st.values())), "flops_by_stage": st, "dominant": max(st, key=st.get),
            "low_memory": bool(low_memory)}


def fit_cugenie_k(geno, y, covariates=None, *, snps=None, offset="loo", lam_grid="wide",
                  gram_dtype="float64", low_memory="auto", null_tol=None, chromosomes=None, device="auto",
                  memory_budget=None, estimate_only=False, block_size=None, y_original=None,
                  prefetch=2, verbose=True):
    """Kernel-ridge LOCO background on CPU or GPU. Returns LocoPredictions (dict for "both").

    snps       : kernel panel (e.g. the PLINK .prune.in list; the CPU-friendly choice); None = all
    lam_grid   : "wide" (default; 1e-7..1e3, 101 points, reproduces the validated reference) |
                 "stable" (1e-5..1e3, 81 points: LOO-CV choices that do not depend on rounding) |
                 an array of lambdas
    offset     : "loo" (default) | "insample" | "both"
    gram_dtype : "float64" (default) or "float32" for the per-block Gram products
    low_memory : False (keep all S_c, ~C n^2 words) | True (totals only, S_c recomputed per
                 chromosome: ~6 n^2 words, twice the reading) | "auto" (False if it fits)
    null_tol   : None (default; eigenvalues clipped at 0, as cugenie_k_v2.py) or a relative
                 tolerance: eigenvalues below null_tol * max eigenvalue are set to exactly 0.
                 K_-c is exactly singular along the covariate directions; at very small lambda
                 the rounding noise of those eigenvalues (~1e-14) enters 1 - diag H = O(lambda)
                 and makes the LOO-CV curve implementation-dependent. 1e-10 removes that.
    chromosomes: chromosomes to use; the LOCO columns are exactly these (default: the panel's
                 chromosomes, columns 1..max so column c-1 = chromosome c)
    """
    if offset not in ("loo", "insample", "both"):
        raise ValueError("offset must be 'loo', 'insample' or 'both'")
    if gram_dtype not in ("float64", "float32"):
        raise ValueError("gram_dtype must be 'float64' or 'float32'")
    dev = _device(device)
    with dev:
        xp = dev.xp
        t_all = time.time()
        T = {}
        Y, names = as_matrix(y, check_binary=True)
        n, P = Y.shape
        if n != geno.n:
            raise ValueError(f"y has {n} rows, genotypes have {geno.n} samples")
        rows = geno.select(snps, chromosomes=chromosomes)
        M = len(rows)
        present = sorted({c for c, _, _ in geno.chrom_ranges(rows)})
        out_chroms = output_chromosomes(present, chromosomes)
        cidx = {c: j for j, c in enumerate(out_chroms)}
        nC = len(out_chroms)
        if lam_grid is None:
            lam_grid = "wide"
        if isinstance(lam_grid, str):
            if lam_grid not in LAM_GRIDS:
                raise ValueError(f"lam_grid must be one of {list(LAM_GRIDS)} or an array of lambdas")
            lam = LAM_GRIDS[lam_grid]
        else:
            lam = np.asarray(lam_grid, dtype=np.float64)
        bud = R.budget(dev, memory_budget)
        bsz = block_size or _auto_block(n, bud, geno.n_file)
        ests = {lm: estimate_cugenie_k(n, M, len(present), P, len(lam), lm, gram_dtype, bsz, geno.n_file)
                for lm in (False, True)}
        host = 0                  # free memory already excludes what is resident now
        if low_memory == "auto":
            lm = not (ests[False]["peak_bytes"] <= bud)
        else:
            lm = bool(low_memory)
        sugg = []
        if not lm:
            sugg.append(f"low_memory=True (estimated peak {R.format_bytes(ests[True]['peak_bytes'] + host)})")
        if gram_dtype == "float64":
            sugg.append("gram_dtype='float32'")
        sugg += ["a pruned kernel panel (snps=...prune.in) cuts the pass time, not the n x n memory",
                 "fit_cugenie (REGENIE-style, blockwise; memory grows with M/block, not n^2)"]
        est = R.finish(ests[lm], dev, memory_budget, sugg, "fit_cugenie_k", geno)
        est["block_size"] = bsz
        if estimate_only:
            return est
        R.check(est, "fit_cugenie_k")
        _log(verbose, dev.describe(), f"| n={n} P={P} M={M:,} low_memory={lm} block={bsz} | est. peak {est['peak']}")

        Qn, _ = covariate_basis(covariates, n)
        ncov = Qn.shape[1]
        Ys, scale_Y, yhat_cov = residualize_scale(Y, Qn)
        Q = dev.asarray(Qn)
        ks = KernelSet(geno, rows, dev, "regenie", Q=Q, ncov=ncov, pairwise_count=False, low_memory=lm,
                       dtype=gram_dtype, block_size=bsz, prefetch=prefetch, verbose=verbose).build()
        T["kernel_pass_s"] = ks.timing["kernel_pass_s"]
        _log(verbose, f"kernel pass {T['kernel_pass_s']:.1f}s, M used {ks.Mtot:,} (low-variance dropped {ks.n_lowvar})")

        t0 = time.time()
        lam_g = dev.asarray(lam)
        Ysg = dev.asarray(Ys)
        hQ = (Q * Q).sum(1)
        L = len(lam)
        off_in = np.zeros((n, nC, P))
        off_loo = np.zeros((n, nC, P))
        curves = np.full((nC, L, P), np.nan)
        lam_star = np.full((nC, P), np.nan)
        t_eig = 0.0
        for c in present:
            K = ks.loco(c)
            ks.drop(c) if not lm else None
            te = time.time()
            d, U = xp.linalg.eigh(K)
            del K
            d = xp.maximum(d, 0.0) if null_tol is None else xp.where(d > null_tol * d.max(), d, 0.0)
            dev.sync()
            t_eig += time.time() - te
            F = d[:, None] / (d[:, None] + lam_g[None, :])
            UtY = U.T @ Ysg
            HY = (U @ (F[:, :, None] * UtY[:, None, :]).reshape(n, -1)).reshape(n, L, P)
            dH = (U * U) @ F + hQ[:, None]
            Rr = Ysg[:, None, :] - HY
            Rloo = Rr / (1.0 - dH)[:, :, None]
            cvm = (Rloo ** 2).mean(0)
            j = dev.asnumpy(xp.argmin(cvm, axis=0))
            jc = cidx[c]
            for k in range(P):
                off_in[:, jc, k] = dev.asnumpy(HY[:, j[k], k])
                off_loo[:, jc, k] = dev.asnumpy(Ysg[:, k] - Rloo[:, j[k], k])
            curves[jc] = dev.asnumpy(cvm)
            lam_star[jc] = lam[j]
            del U, d, F, HY, dH, Rr, Rloo
        T["eigh_s"] = t_eig
        T["loocv_offsets_s"] = time.time() - t0 - t_eig
        T["kernel_recompute_s"] = ks.timing["kernel_recompute_s"]
        ks_M = dict(ks.M)
        del ks
        dev.free_pool()
        T["total_s"] = time.time() - t_all
        r2_loo = 1 - ((Ys[:, None, :] - off_loo) ** 2).mean(0) / (Ys ** 2).mean(0)[None, :]
        meta = {"n": n, "ncov": ncov, "M": M, "low_memory": lm, "gram_dtype": gram_dtype, "null_tol": null_tol,
                "M_per_chr": {int(c): int(m) for c, m in ks_M.items()},
                "lam_grid": lam.tolist(), "lam_star": lam_star.tolist(), "h2_star": (1 / (1 + lam_star)).tolist(),
                "loocv_curves": curves, "r2_loo": r2_loo.tolist(), "scale_y": scale_Y.tolist(),
                "counted_allele": geno.counted_label, "device": str(dev), "timing_s": T,
                "chromosomes": list(out_chroms)}
        pj = np.array([cidx[c] for c in present], dtype=np.int64)
        ls = lam_star[pj]
        n_low = int((ls < LAM_STABLE).sum())
        meta["lam_stable_floor"] = LAM_STABLE
        meta["n_lam_star_below_stable_floor"] = n_low
        if n_low:
            logger.warning(
                f"lambda* < {LAM_STABLE:g} for {n_low} of {ls.size} (chromosome, phenotype) fits "
                f"(h2* -> 1, near interpolation). There 1 - diag H = O(lambda) and the LOO-CV curve is dominated "
                f"by rounding, so lambda* (not the offsets, which move by ~1e-6) is implementation-dependent. "
                f"lam_grid='stable' (1e-5..1e3) gives implementation-stable choices; UltraLOCO is the calibrated "
                f"option in this regime.")
        for k, nm in enumerate(names):
            h = 1 / (1 + lam_star[pj, k])
            _log(verbose, f"{nm}: h2* median {np.median(h):.3f} [{h.min():.3f}, {h.max():.3f}], LOO R2 median "
                 f"{np.nanmedian(r2_loo[pj, k]):.3f}")
        _log(verbose, "timings", {k: round(v, 2) for k, v in T.items()})
        yo = Y if y_original is None else as_matrix(y_original, "y_original")[0]
        res = {}
        for flav, off in (("loo", off_loo), ("insample", off_in)):
            if offset in (flav, "both"):
                res[flav] = _build_loco(off, scale_Y, yhat_cov, yo, geno.sample_ids, names, f"cugenie_k_{flav}", meta,
                                        out_chroms)
        return res if offset == "both" else res[offset]
