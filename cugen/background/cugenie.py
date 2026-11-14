"""CuGenie: REGENIE-style step 1 (block ridge level 0 + ridge stacking level 1, K-fold CV, LOCO).

Validated against REGENIE v2.2.4 (tests/test_regenie_gold.py and public Darwin's Ark data):
level-1 CV R2/MSE equal to every printed digit, masked LOCO offsets r = 1.000000 with
max|diff| 5e-6 (REGENIE's text precision).
Genotypes are streamed block by block (never resident) and the same code runs on numpy or
cupy through ``device``.

REGENIE details reproduced (v2.2.4 source):
  * phenotypes y - QQ'y scaled by ||.||/sqrt(n-C); genotypes mean-imputed, residualised and
    scaled the same way; blocks = ceil(m_c / block_size) contiguous blocks per chromosome;
  * folds contiguous in sample order (``fold_kind="contiguous"``), floor(n/K) each;
  * level 0: lambda = M (1-h2)/h2, M = all step-1 variants; out-of-fold predictions, each
    (block, ridge) column centred and scaled over the whole sample;
  * level 1: tau = B R (1-h2)/h2, chosen by pooled out-of-fold MSE;
  * ``loco="masked"`` (default, REGENIE-exact): the fold-i level-1 fit with chromosome c's
    coefficients dropped, no refit. ``loco="refit"``: per chromosome the level-1 ridge is
    refit on the other chromosomes' predictors (own tau by CV). ``"both"`` returns both.

Sample order matters for REGENIE equivalence: REGENIE's folds are contiguous in .fam order,
so pass samples in .fam order (``BedGenotypes(samples=None)`` or a .fam-ordered list).
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .. import _resources as R
from .._device import device as _device
from ._prep import as_matrix, covariate_basis, fold_index, residualize_scale, standardize_regenie
from .loco import build as _build_loco
from .loco import output_chromosomes

logger = logging.getLogger("cugen.background.cugenie")

H2_GRID = (0.01, 0.25, 0.5, 0.75, 0.99)


def _log(verbose, *a):
    if verbose:
        logger.info(" ".join(str(x) for x in a))


def estimate_cugenie(n, M, n_blocks, P=1, block_size=1000, folds=5, R0=5, R1=5, loco="masked",
                     n_chrom=38, n_file=None, itemsize=8):
    """Peak bytes / FLOPs of fit_cugenie (genotypes streamed)."""
    n_file = n_file or n
    bs = min(block_size, M)
    p1 = n_blocks * R0
    W = P * n * p1 * itemsize
    blk = bs * n * itemsize * 3 + bs * ((n_file + 3) // 4) * 4 * 3          # G, temporaries, decode
    l0 = (folds + 2) * bs * bs * itemsize + bs * R0 * P * 4 * itemsize + n * R0 * P * itemsize
    l1 = (folds + 4) * p1 * p1 * itemsize + 2 * W
    off = 3 * n * n_chrom * P * 8
    peak = int(W + max(blk + l0, l1) + off + 64e6)
    fl0 = n_blocks * (2 * bs * bs * n + folds * (10 * bs ** 3 + 4 * bs * bs * R0 * P))
    fl1 = P * (2 * p1 * p1 * n + folds * 10 * p1 ** 3)
    flr = (P * n_chrom * (2 * p1 * p1 * n + folds * 10 * p1 ** 3)) if loco in ("refit", "both") else 0
    stages = {"level0": fl0, "level1": fl1, "refit": flr}
    dom = max(stages, key=stages.get)
    return {"method": "cugenie", "n": n, "M": M, "n_blocks": n_blocks, "peak_bytes": peak,
            "peak_breakdown": {"level0_predictors_W": W, "block": blk + l0, "level1": l1},
            "flops": float(sum(stages.values())), "flops_by_stage": stages, "dominant": dom}


def fit_cugenie(geno, y, covariates=None, *, snps=None, loco="masked", block_size=1000, folds=5,
                fold_kind="contiguous", seed=0, h2_l0=H2_GRID, h2_l1=H2_GRID, chromosomes=None,
                device="auto", memory_budget=None, estimate_only=False, y_original=None,
                prefetch=2, verbose=True):
    """REGENIE-style step 1 on CPU or GPU. Returns LocoPredictions (dict for loco="both").

    geno        : BedGenotypes / CugenGenotypes (rows of y = geno.sample_ids order)
    y           : (n,) or (n, P) quantitative phenotype(s), complete cases
    covariates  : (n, k) or None; an intercept is always added
    snps        : step-1 panel (IDs, path to .snplist/.prune.in, mask or rows); None = all
    loco        : "masked" (REGENIE-exact, default) | "refit" (per-chromosome level-1 refit on
                  the other chromosomes, own tau by CV) | "both"
    block_size  : level-0 block size (REGENIE --bsize); folds / fold_kind: K and "contiguous"
                  (REGENIE, in sample order) | "random" (seeded)
    h2_l0/h2_l1 : heritability grids of the level-0 / level-1 ridges (REGENIE's defaults)
    chromosomes : chromosomes to use; the LOCO columns are exactly these (default: the panel's
                  chromosomes, columns 1..max so column c-1 = chromosome c)
    device      : "auto" | "cpu" | "cuda" | "cuda:N" | int
    estimate_only / memory_budget : see :mod:`cugen._resources`
    y_original  : phenotype on the scale to store in the predictions (default: y)
    """
    if loco not in ("masked", "refit", "both"):
        raise ValueError("loco must be 'masked', 'refit' or 'both'")
    dev = _device(device)
    with dev:
        xp = dev.xp
        T = {}
        t_all = time.time()
        Y, names = as_matrix(y, check_binary=True)
        n, P = Y.shape
        if n != geno.n:
            raise ValueError(f"y has {n} rows, genotypes have {geno.n} samples")
        rows = geno.select(snps, chromosomes=chromosomes)
        M = len(rows)
        blocks = geno.blocks(rows, block_size)
        B = len(blocks)
        h2_0 = np.asarray(h2_l0, dtype=np.float64)
        h2_1 = np.asarray(h2_l1, dtype=np.float64)
        R0, R1 = len(h2_0), len(h2_1)
        present = sorted({b[0] for b in blocks})
        out_chroms = output_chromosomes(present, chromosomes)
        cidx = {c: j for j, c in enumerate(out_chroms)}
        nC = len(out_chroms)
        est = estimate_cugenie(n, M, B, P, block_size, folds, R0, R1, loco, nC, geno.n_file)
        est = R.finish(est, dev, memory_budget, [
            f"block_size={max(block_size // 2, 100)} (smaller level-0 blocks)",
            "a smaller step-1 panel (snps=pruned list) reduces the level-0 predictor matrix",
            "loco='masked' (no per-chromosome refit)" if loco != "masked" else
            "fewer phenotypes per call"], "fit_cugenie", geno)
        if estimate_only:
            return est
        R.check(est, "fit_cugenie")
        if not getattr(geno, "sample_order_is_file_order", True) and fold_kind == "contiguous":
            _log(verbose, "note: sample order is not file order; contiguous folds differ from REGENIE's")
        _log(verbose, dev.describe(), f"| n={n} P={P} M={M:,} blocks={B} | est. peak {est['peak']}")

        Qn, _ = covariate_basis(covariates, n)
        ncov = Qn.shape[1]
        Ys, scale_Y, yhat_cov = residualize_scale(Y, Qn)
        fidx = fold_index(n, folds, fold_kind, seed)
        contiguous = fold_kind == "contiguous"
        lam0 = M * (1 - h2_0) / h2_0
        tau = (B * R0) * (1 - h2_1) / h2_1
        bchr = np.array([b[0] for b in blocks])
        col_chr = np.repeat(bchr, R0)
        meta = {"n": n, "ncov": ncov, "M": M, "n_blocks": B, "folds": [len(f) for f in fidx],
                "fold_kind": fold_kind, "lambda_l0": lam0.tolist(), "tau_l1": tau.tolist(),
                "scale_y": scale_Y.tolist(), "counted_allele": geno.counted_label, "device": str(dev),
                "chromosomes": list(out_chroms)}

        # ---------------- level 0 (streamed) ----------------
        t0 = time.time()
        Q = dev.asarray(Qn)
        Ysg = dev.asarray(Ys)
        lam_g = dev.asarray(lam0)
        W = xp.zeros((P, n, B * R0), dtype=xp.float64)
        fs = [slice(int(f[0]), int(f[-1]) + 1) for f in fidx] if contiguous else [dev.asarray(f) for f in fidx]
        n_lowvar = 0
        for b, (c, s, e, counts) in enumerate(geno.iter_blocks(rows, blocks, dev, prefetch)):
            Gb, ok = standardize_regenie(counts, Q, ncov, xp)
            del counts
            nbad = int(dev.asnumpy((~ok).sum()))
            n_lowvar += nbad
            bs = e - s
            Gf = [Gb[:, f] @ Gb[:, f].T for f in fs]
            GtYf = [Gb[:, f] @ Ysg[f] for f in fs]
            GGt = sum(Gf)
            GTY = sum(GtYf)
            for i, f in enumerate(fs):
                d, V = xp.linalg.eigh(GGt - Gf[i])
                ww2 = V.T @ (GTY - GtYf[i])                                            # (bs, P)
                beta = (V @ (ww2[:, None, :] / (d[:, None, None] + lam_g[None, :, None])).reshape(bs, R0 * P)
                        ).reshape(bs, R0, P)
                Gfi = Gb[:, f]
                ni = Gfi.shape[1]
                pred = (Gfi.T @ beta.reshape(bs, R0 * P)).reshape(ni, R0, P).transpose(2, 0, 1)
                W[:, f, b * R0:(b + 1) * R0] = pred
            del Gb, Gf, GtYf, GGt
        dev.sync()
        mu = W.mean(1, keepdims=True)
        sd = xp.sqrt(((W - mu) ** 2).sum(1, keepdims=True) / (n - 1))
        W -= mu
        W /= sd
        del mu, sd
        T["level0_s"] = time.time() - t0
        meta["n_lowvar_zeroed"] = n_lowvar
        if n_lowvar:
            logger.warning("%d low-variance step-1 SNPs were zeroed (REGENIE would stop)", n_lowvar)
        _log(verbose, f"level 0: {B} blocks x {folds} folds, {T['level0_s']:.1f}s")

        # ---------------- level 1 ----------------
        def level1(Wp, yv, cols, tau_v):
            Xf = [Wp[f][:, cols].T @ Wp[f][:, cols] for f in fs]
            Xy = [Wp[f][:, cols].T @ yv[f] for f in fs]
            XS = sum(Xf)
            XyS = sum(Xy)
            betas = []
            sx = np.zeros(R1); sx2 = np.zeros(R1); sxy = np.zeros(R1); sy = 0.0; sy2 = 0.0
            tg = dev.asarray(tau_v)
            for i, f in enumerate(fs):
                d, V = xp.linalg.eigh(XS - Xf[i])
                vt = V.T @ (XyS - Xy[i])
                bet = V @ (vt[:, None] / (d[:, None] + tg[None, :]))
                betas.append(bet)
                p1 = Wp[f][:, cols] @ bet
                yi = yv[f]
                sx += dev.asnumpy(p1.sum(0)); sy += float(yi.sum()); sx2 += dev.asnumpy((p1 ** 2).sum(0))
                sy2 += float((yi ** 2).sum()); sxy += dev.asnumpy((p1 * yi[:, None]).sum(0))
            mse = (sx2 + sy2 - 2 * sxy) / n
            num = sxy - sx * sy / n
            rsq = num ** 2 / ((sx2 - sx ** 2 / n) * (sy2 - sy ** 2 / n))
            return betas, mse, rsq, int(np.argmin(mse))

        out = {}
        t0 = time.time()
        if loco in ("masked", "both"):
            Lm = np.zeros((n, nC, P))
            l1 = {}
            ind = np.zeros((B * R0, nC))
            ind[np.arange(B * R0), np.array([cidx[c] for c in col_chr], dtype=np.int64)] = 1.0
            ind_g = dev.asarray(ind)
            allcols = xp.arange(B * R0)
            for k in range(P):
                Wp = W[k]
                yv = Ysg[:, k]
                betas, mse, rsq, j = level1(Wp, yv, allcols, tau)
                l1[names[k]] = {"mse": mse.tolist(), "rsq": rsq.tolist(), "tau_index": j, "h2_l1": float(h2_1[j])}
                for i, f in enumerate(fs):
                    contrib = (Wp[f] * betas[i][:, j][None, :]) @ ind_g
                    tot = contrib.sum(1)
                    idx = fidx[i]
                    Lm[idx, :, k] = dev.asnumpy(tot[:, None] - contrib)
                _log(verbose, f"level 1 {names[k]}: " + "  ".join(
                    f"h2={h:.2f} Rsq={r:.6f} MSE={m:.6f}{'<-' if q == j else ''}"
                    for q, (h, r, m) in enumerate(zip(h2_1, rsq, mse))))
            meta["level1"] = l1
            T["level1_masked_s"] = time.time() - t0
            out["masked"] = Lm
        if loco in ("refit", "both"):
            t0 = time.time()
            Lr = np.zeros((n, nC, P))
            choice = np.zeros((nC, P), dtype=int)
            for k in range(P):
                Wp = W[k]
                yv = Ysg[:, k]
                for c in present:
                    cols = dev.asarray(np.flatnonzero(col_chr != c))
                    Bc = int((bchr != c).sum())
                    tau_c = (Bc * R0) * (1 - h2_1) / h2_1
                    betas, mse, rsq, j = level1(Wp, yv, cols, tau_c)
                    choice[cidx[c], k] = j
                    for i, f in enumerate(fs):
                        Lr[fidx[i], cidx[c], k] = dev.asnumpy(Wp[f][:, cols] @ betas[i][:, j])
            meta["refit_tau_index_per_chr"] = choice.tolist()
            T["level1_refit_s"] = time.time() - t0
            out["refit"] = Lr
        del W
        dev.free_pool()
        T["total_s"] = time.time() - t_all
        meta["timing_s"] = T
        _log(verbose, "timings", {k: round(v, 2) for k, v in T.items()})
        yo = Y if y_original is None else as_matrix(y_original, "y_original")[0]
        res = {v: _build_loco(L, scale_Y, yhat_cov, yo, geno.sample_ids, names, f"cugenie_{v}", meta, out_chroms)
               for v, L in out.items()}
        return res if loco == "both" else res[loco]
