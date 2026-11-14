"""UltraLOCO: GLS-LOCO mixed-model association (GCTA-MLMA-LOCO equivalent), CPU or GPU.

Validated against a dense GLS (max dZ 1.2e-6) and against GCTA-MLMA-LOCO on public Darwin's Ark data:

  * kernel pass: per-chromosome GRM numerators S_c = sum z z' with GCTA
    standardisation z = (g - 2p)/sqrt(2p(1-p)), missing -> 0, and pairwise valid-marker counts
    N_c (``pairwise_count=True``, GCTA style) or a plain marker count;
  * per chromosome: K_-c = (S - S_c)/(N - N_c), eigh, profile REML over
    delta = se2/sg2 on the eigenbasis (61-point log grid on [1e-3, 1e4] + golden section).
    Negative eigenvalues of K_-c (a pairwise-count GRM need not be PSD) are
    clipped to 0, i.e. the GLS uses V = K_-c,+ + delta I with K_-c,+ the PSD part;
  * whitened test: x mean-imputed and centred, beta = x'Py / x'Px,
    SE = sqrt(sg2 / x'Px), Z = beta/SE.

``scaling="regenie"`` builds the kernel from CuGenie-K's standardisation instead (covariates
projected out, mean-imputed, ||z||^2 = n - ncov, divided by the marker count): CuGenie-K
kernels + REML = the MLMA-whitened CuGenie-K.
"""
from __future__ import annotations

import logging
import time

import numpy as np

from .. import _resources as R
from .._assoc_blocked import _annotation_columns, _auto_block_step4, _frame, _str_bytes_per_row
from .._device import device as _device
from ._kernels import KernelSet, kernel_bytes
from ._prep import as_matrix, covariate_basis, design

logger = logging.getLogger("cugen.background.ultraloco")


def _log(verbose, *a):
    if verbose:
        logger.info(" ".join(str(x) for x in a))


def reml(s, ys, Xs, xp, dmin=1e-3, dmax=1e4, ngrid=61):
    """Profile REML over delta = se2/sg2 on the eigenbasis of K.
    Returns (delta, sg2, loglik)."""
    n, k = Xs.shape
    sp = xp.maximum(s, 0)

    def ll(ld):
        d = 10.0 ** ld
        w = 1.0 / (sp + d)
        A = (Xs * w[:, None]).T @ Xs
        b = (Xs * w[:, None]).T @ ys
        yPy = float((w * ys * ys).sum() - b @ xp.linalg.solve(A, b))
        sign, logdetA = xp.linalg.slogdet(A)
        return -0.5 * ((n - k) * np.log(yPy / (n - k)) + float(xp.log(sp + d).sum()) + float(logdetA)), yPy

    grid = np.linspace(np.log10(dmin), np.log10(dmax), ngrid)
    vals = [ll(g)[0] for g in grid]
    i = int(np.argmax(vals))
    lo, hi = grid[max(i - 1, 0)], grid[min(i + 1, len(grid) - 1)]
    gr = (np.sqrt(5) - 1) / 2
    a, b = lo, hi
    c, d = b - gr * (b - a), a + gr * (b - a)
    for _ in range(40):
        if ll(c)[0] > ll(d)[0]:
            b = d
        else:
            a = c
        c, d = b - gr * (b - a), a + gr * (b - a)
    ld = (a + b) / 2
    if vals[0] >= ll(ld)[0]:
        ld = grid[0]
    val, yPy = ll(ld)
    return 10.0 ** ld, yPy / (n - k), val


def estimate_ultraloco(n, M_grm, M_test, n_chrom=38, pairwise_count=True, low_memory=False,
                       block_size=8192, n_file=None, test_itemsize=4, str_bytes_per_row=24):
    """Kernels stay resident through the association; per-variant results accumulate (32 B/row) during
    the test pass; the result assembly needs ~(113 + strings) B/row (see step4.estimate_step4)."""
    n_file = n_file or n
    block_size = max(1, min(block_size, max(M_grm, M_test)))
    kern = kernel_bytes(n, n_chrom, pairwise_count, low_memory, work=8)
    blk = block_size * n * (8 * 3 + 2 * test_itemsize) + block_size * ((n_file + 3) // 4) * 4 * 2
    out = int(M_test * (113 + str_bytes_per_row))
    peak = int(kern + max(blk + M_test * 32, out) + 64e6)
    per = 2 if pairwise_count else 1
    st = {"kernel_pass": 2.0 * M_grm * n * n * per * (2 if low_memory else 1),
          "eigh_reml": n_chrom * 12.0 * n ** 3,
          "test": 2.0 * M_test * n * n}
    return {"method": "ultraloco", "n": n, "M_grm": M_grm, "M_test": M_test, "peak_bytes": peak,
            "peak_breakdown": {"kernels": kern, "test_block": blk, "output": out},
            "flops": float(sum(st.values())), "flops_by_stage": st, "dominant": max(st, key=st.get),
            "low_memory": bool(low_memory)}


class UltraLocoModel:
    """Kernel state + phenotype/covariates; ``gwas()`` runs the per-chromosome REML + test."""

    def __init__(self, geno, y, X0, kernels, dev, meta, verbose, chromosomes=None):
        self.geno, self.y, self.X0, self.kernels, self.dev = geno, y, X0, kernels, dev
        self.meta = meta
        self.verbose = verbose
        self.chromosomes = None if chromosomes is None else sorted({int(c) for c in chromosomes})
        self.chrom_info = {}

    def gwas(self, **kw):
        """The per-chromosome REML + whitened test; keyword arguments as :func:`ultraloco_gwas`."""
        return ultraloco_gwas(self, **kw)

    def __repr__(self):
        return (f"UltraLocoModel(n={self.geno.n}, M_grm={self.meta.get('M_grm')}, "
                f"scaling={self.meta.get('scaling')!r}, device={self.meta.get('device')!r})")


def fit_ultraloco(geno, y, covariates=None, *, grm_snps=None, pairwise_count=True, scaling="gcta",
                  low_memory="auto", gram_dtype="float64", chromosomes=None, device="auto",
                  memory_budget=None, estimate_only=False, block_size=None, test_snps=None,
                  prefetch=2, verbose=True):
    """Kernel pass for GLS-LOCO. Returns an UltraLocoModel; call ``.gwas()`` (or
    ``ultraloco_gwas(model)``) for the association.

    grm_snps       : GRM panel (IDs / path / mask / rows); None = all variants. A pruned panel
                     is the CPU-friendly choice (the pass is 2 M n^2 flops per matrix).
    pairwise_count : GCTA-style elementwise marker-count normalisation (default) or M_-c.
    scaling        : "gcta" (default) or "regenie" (CuGenie-K kernels; implies pairwise_count=False)
    low_memory     : False (keep S_c, N_c: ~2 C n^2 words) | True (totals only; per-chromosome
                     recompute) | "auto" (False if it fits)
    gram_dtype     : "float64" (default) or "float32" for the per-block Gram products
    chromosomes    : chromosomes for the GRM and (by default) the association
    test_snps      : variants the association will test (used by ``estimate_only``)
    estimate_only  : returns the estimate for the kernel pass + association (test_snps)

    ``scaling="regenie"`` with ``pairwise_count=False`` is the MLMA-whitened CuGenie-K: the
    same kernel as :func:`fit_cugenie_k`, used for GLS whitening instead of an offset.
    """
    if scaling not in ("gcta", "regenie"):
        raise ValueError("scaling must be 'gcta' or 'regenie'")
    if scaling == "regenie":
        pairwise_count = False
    dev = _device(device)
    with dev:
        Y, names = as_matrix(y, check_binary=True)
        if Y.shape[1] != 1:
            raise ValueError("UltraLOCO fits one phenotype at a time")
        n = Y.shape[0]
        if n != geno.n:
            raise ValueError(f"y has {n} rows, genotypes have {geno.n} samples")
        rows = geno.select(grm_snps, chromosomes=chromosomes)
        trows = geno.select(test_snps, chromosomes=chromosomes) if test_snps is not None else None
        if trows is not None:
            M_test = len(trows)
        elif chromosomes is not None:
            M_test = int(np.isin(geno.chrom, np.asarray(list(chromosomes), dtype=np.int64)).sum())
        else:
            M_test = geno.n_variants
        present = sorted({c for c, _, _ in geno.chrom_ranges(rows)})
        bud = R.budget(dev, memory_budget)
        bsz = block_size or _auto_block_step4(geno.n, bud, geno.n_file, dev)
        sbr = _str_bytes_per_row(geno)
        ests = {lm: estimate_ultraloco(n, len(rows), M_test, len(present), pairwise_count, lm, bsz, geno.n_file,
                                       str_bytes_per_row=sbr)
                for lm in (False, True)}
        host = 0                  # free memory already excludes what is resident now
        lm = (not (ests[False]["peak_bytes"] <= bud)) if low_memory == "auto" else bool(low_memory)
        sugg = []
        if not lm:
            sugg.append(f"low_memory=True (estimated peak {R.format_bytes(ests[True]['peak_bytes'] + host)})")
        if pairwise_count:
            sugg.append("pairwise_count=False (halves the kernel memory and the pass)")
        sugg += ["a smaller test block (block_size=...)",
                 "fit_cugenie + background.gwas_blocked (REGENIE-style; no n x n matrices per chromosome)"]
        est = R.finish(ests[lm], dev, memory_budget, sugg, "fit_ultraloco", geno)
        est["block_size"] = bsz
        if estimate_only:
            return est
        R.check(est, "fit_ultraloco")
        _log(verbose, dev.describe(), f"| n={n} M_grm={len(rows):,} pairwise={pairwise_count} scaling={scaling} "
             f"low_memory={lm} | est. peak {est['peak']}")
        Q = None
        ncov = 0
        if scaling == "regenie":
            Qn, _ = covariate_basis(covariates, n)
            Q, ncov = dev.asarray(Qn), Qn.shape[1]
        ks = KernelSet(geno, rows, dev, scaling, Q=Q, ncov=ncov, pairwise_count=pairwise_count,
                       low_memory=lm, dtype=gram_dtype, block_size=bsz, prefetch=prefetch,
                       verbose=verbose).build()
        _log(verbose, f"kernel pass {ks.timing['kernel_pass_s']:.1f}s, markers {ks.Mtot:,}")
        meta = {"n": n, "M_grm": ks.Mtot, "pairwise_count": pairwise_count, "scaling": scaling,
                "low_memory": lm, "gram_dtype": gram_dtype, "phenotype": names[0],
                "counted_allele": geno.counted_label, "device": str(dev), "block_size": bsz,
                "timing_s": {"kernel_pass_s": ks.timing["kernel_pass_s"]}}
        X0 = design(covariates, n)
        return UltraLocoModel(geno, Y[:, 0], X0, ks, dev, meta, verbose, chromosomes)


def ultraloco_gwas(model, y=None, covariates=None, *, snps=None, maf_min=1e-4, block_size=None,
                   test_dtype="float32", chromosomes=None, reml_bounds=(1e-3, 1e4), reml_grid=61,
                   prefetch=2, **fit_kw):
    """UltraLOCO association: per chromosome, eigh of K_-c, REML, whitened test of its variants.

    Two forms, mirroring :func:`cugen.ultralasso_gwas`:

        model = fit_ultraloco(geno, y, covariates, ...); ss = ultraloco_gwas(model)   # or model.gwas()
        ss = ultraloco_gwas(geno, y, covariates, grm_snps=..., ...)                   # end to end

    In the end-to-end form ``fit_kw`` go to :func:`fit_ultraloco` (``grm_snps``,
    ``pairwise_count``, ``scaling``, ``low_memory``, ``gram_dtype``, ``device``,
    ``memory_budget``, ``verbose``); ``chromosomes``, ``block_size`` and ``prefetch`` are
    used by both stages.

    snps        : variants to test (IDs / path / mask / rows); None = all
    maf_min     : drop variants with MAF below this (default 1e-4, as ``cugen.gwas``; None keeps all)
    test_dtype  : "float32" (default) or "float64" for the whitened test products
    chromosomes : chromosomes to test (default: the model's chromosomes, else all)
    reml_bounds : (min, max) of delta = se2/sg2 searched by REML (default 1e-3..1e4).
                  A chromosome whose delta lands on a bound is flagged (``at_bound``) and logged.
    Returns a sumstats DataFrame with the step-4 columns ``CHR POS ID REF ALT BETA SE Z P MAF
    gidx`` (BETA per copy of the counted allele, ALT = counted allele, named in
    ``df.attrs['counted_allele']``); per-chromosome REML results in ``df.attrs['chrom_info']``."""
    if not isinstance(model, UltraLocoModel):
        if y is None:
            raise TypeError("ultraloco_gwas(geno, y, covariates, ...) needs y; or pass a fitted UltraLocoModel")
        model = fit_ultraloco(model, y, covariates, chromosomes=chromosomes, block_size=block_size,
                              prefetch=prefetch, **fit_kw)
    elif y is not None or covariates is not None or fit_kw:
        raise TypeError("y / covariates / fit options are only accepted with genotypes; the model is "
                        "already fitted")
    if chromosomes is None:
        chromosomes = model.chromosomes
    with model.dev:
        return _ultraloco_test(model, snps=snps, maf_min=maf_min, block_size=block_size,
                               test_dtype=test_dtype, chromosomes=chromosomes, reml_bounds=reml_bounds,
                               reml_grid=reml_grid, prefetch=prefetch)


def _ultraloco_test(model, *, snps, maf_min, block_size, test_dtype, chromosomes, reml_bounds,
                    reml_grid, prefetch):
    geno, dev, ks = model.geno, model.dev, model.kernels
    xp = dev.xp
    verbose = model.verbose
    n = geno.n
    tdt = np.dtype(test_dtype)
    rows = geno.select(snps, chromosomes=chromosomes)
    bsz = block_size or model.meta["block_size"]
    y = dev.asarray(model.y)
    X0 = dev.asarray(model.X0)
    t0 = time.time()
    t_eig = t_test = 0.0
    parts = []
    tranges = {c: (s, e) for c, s, e in geno.chrom_ranges(rows)}
    for c in sorted(tranges):
        te = time.time()
        Km = ks.loco(c)
        s_, U = xp.linalg.eigh(Km)
        del Km
        ys = U.T @ y
        Xs = U.T @ X0
        delta, sg2, llv = reml(s_, ys, Xs, xp, dmin=reml_bounds[0], dmax=reml_bounds[1], ngrid=reml_grid)
        at_bound = bool(delta <= reml_bounds[0] * 1.0001 or delta >= reml_bounds[1] / 1.0001)
        w = 1.0 / (xp.maximum(s_, 0) + delta)
        WXs = Xs * w[:, None]
        Ainv = xp.linalg.inv(Xs.T @ WXs)
        b0 = WXs.T @ ys
        Py = w * ys - WXs @ (Ainv @ b0)
        model.chrom_info[c] = {"delta": float(delta), "h2": float(1 / (1 + delta)), "sg2": float(sg2),
                               "reml_ll": float(llv), "min_eig": float(s_.min()), "at_bound": at_bound}
        dev.sync()
        t_eig += time.time() - te
        te = time.time()
        Uf = U.astype(tdt)
        Pyf = Py.astype(tdt)
        WXsf = WXs.astype(tdt)
        wf = w.astype(tdt)
        s0, e0 = tranges[c]
        blocks = [(c, s, min(s + bsz, e0)) for s in range(s0, e0, bsz)]
        nv = e0 - s0
        Bh = np.empty(nv, np.float64); Sh = np.empty(nv, np.float64); Mh = np.empty(nv, np.float64)
        lut = xp.asarray(np.array([0.0, 1.0, 2.0, 0.0]), dtype=tdt)
        for _, s, e, counts in geno.iter_blocks(rows, blocks, dev, prefetch):
            miss = counts == 3
            gf = lut[counts]
            nval = (~miss).sum(1).astype(tdt)
            mu = gf.sum(1) / xp.maximum(nval, 1)
            x = xp.where(miss, 0, gf - mu[:, None]).astype(tdt).T          # (n, b)
            del gf, miss, counts
            xs = Uf.T @ x
            xWx = (wf[:, None] * xs * xs).sum(0).astype(xp.float64)
            xWC = (xs.T @ WXsf).astype(xp.float64)
            quad = ((xWC @ Ainv) * xWC).sum(1)
            den = xp.maximum(xWx - quad, 1e-12)
            nm_ = (xs.T @ Pyf).astype(xp.float64)
            bg = nm_ / den
            seg = xp.sqrt(sg2 / den)
            i0, i1 = s - s0, e - s0
            Bh[i0:i1] = dev.asnumpy(bg); Sh[i0:i1] = dev.asnumpy(seg); Mh[i0:i1] = dev.asnumpy(mu)
            del x, xs
        dev.sync()
        t_test += time.time() - te
        del U, Uf, s_
        maf = np.minimum(Mh / 2, 1 - Mh / 2)
        parts.append((rows[s0:e0], Bh, Sh, maf))
        if verbose:
            ci = model.chrom_info[c]
            _log(verbose, f"chr{c}: {nv:,} variants, h2={ci['h2']:.3f}, {time.time() - te:.1f}s test")
    dev.free_pool()
    nb = sum(1 for c in tranges if model.chrom_info[c]["at_bound"])
    if nb:
        logger.warning(f"REML delta at a search bound {reml_bounds} on {nb} of {len(tranges)} chromosomes "
                       f"(h2 -> {1 / (1 + reml_bounds[0]):.4f} at the lower bound); widen reml_bounds to "
                       f"check sensitivity")
    rr = np.concatenate([p[0] for p in parts]); B = np.concatenate([p[1] for p in parts])
    S = np.concatenate([p[2] for p in parts]); maf = np.concatenate([p[3] for p in parts])
    del parts
    import scipy.special as ssp
    if maf_min is not None and maf_min > 0:
        keep = maf >= maf_min
        rr, B, S, maf = rr[keep], B[keep], S[keep], maf[keep]
    Z = B / S
    st_cols = {"BETA": B.astype(np.float32), "SE": S.astype(np.float32), "Z": Z.astype(np.float32),
               "P": ssp.erfc(np.abs(Z) / np.sqrt(2.0)), "MAF": maf.astype(np.float32)}
    del B, S, Z, maf
    front, back = _annotation_columns(geno, rr)
    df = _frame(front, st_cols, back, geno.counted_label)
    model.meta["reml_bounds"] = list(reml_bounds)
    model.meta["n_chrom_reml_at_bound"] = nb
    model.meta["timing_s"].update(eigh_reml_s=t_eig, test_s=t_test, gwas_total_s=time.time() - t0,
                                  kernel_recompute_s=ks.timing["kernel_recompute_s"])
    df.attrs["chrom_info"] = dict(model.chrom_info)
    df.attrs["meta"] = dict(model.meta)
    _log(verbose, f"association {time.time() - t0:.1f}s (eigh+REML {t_eig:.1f}s, test {t_test:.1f}s); "
         f"BETA per copy of the counted allele: {geno.counted_label}")
    return df
