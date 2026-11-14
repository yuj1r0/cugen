"""cugen.datasets._height - synthetic "exaggerated human height" GWAS dataset.

CPU only, numpy only. Produces a fully synthetic, shareable cohort that the
standard UltraLasso workflow (``cg.ultra_workflow``) runs on without glue:
per-chromosome ``.cugen`` files written with ``cugen.write.CugenWriter`` (so the
stored mu_x / sxx / maf are exact), a PLINK1 ``.bed/.bim/.fam`` for REGENIE and
plink, the phenotype / covariate tables, the ground truth and a ready config.

Genotype model (haplotype mosaic)
---------------------------------
* Variants are split over ``n_chrom`` autosomes in proportion to the GRCh37
  autosome lengths (``chrom_lengths="grch37"``, the default for up to 22
  chromosomes) or equally (``"equal"``, the default above 22, e.g. 38 for a
  dog-like genome). Positions are synthetic (build "synthetic"), exponential
  gaps with mean ``spacing_bp`` (2.9 kb default = human genome / 1M variants).
* Each chromosome is cut into LD blocks of random length (gamma, shape 2, mean
  ``block_mean`` variants). Blocks are generated independently, so LD between
  blocks is zero in expectation.
* A block has ``n_founders`` founder haplotypes with block-specific founder
  frequencies w ~ Dirichlet(``founder_conc``). Every sample haplotype copies a
  founder and switches founder inside the block as a Markov chain (switch
  probability per gap 1 - exp(-``switch_rate`` * gap / spacing_bp)), which is
  the recombination that makes LD decay with distance inside a block.
* Founder alleles come from a latent Gaussian AR(1) process along the block
  (correlation exp(-gap / ``ld_scale_bp``) between neighbours) thresholded so
  that the founder-weighted ALT frequency hits the variant's target frequency
  (target MAF ~ truncated 0.5 * Beta(``maf_beta``), MAF >= ``maf_min``). The
  founder straddling the threshold is a "partial" founder: only a random,
  nested subset of its copies carries the allele (a younger mutation), which
  makes the realised frequency continuous instead of quantised in founder
  units. Genotype = sum of the two haplotypes = ALT dosage.

Population structure (``n_pops > 1``)
-------------------------------------
Balding-Nichols applied at the haplotype level: population founder frequencies
w_p ~ Dirichlet(w (1 - F) / F). By the Dirichlet aggregation property the ALT
frequency of every variant is then p_p ~ Beta(p (1 - F) / F, (1 - p)(1 - F) / F),
i.e. exactly the Balding-Nichols model with Fst = ``fst``, while the within-block
LD structure is preserved. ``strat_shift`` adds a population mean shift to the
phenotype (pop means evenly spaced in [-strat_shift, +strat_shift]).

Phenotype
---------
y = g + e + covariates (+ population shift), g = G beta scaled so that the SNP
heritability of the covariate-adjusted trait is exactly ``h2`` in the training
samples. ``n_large`` large-effect loci (common, MAF >= ``large_maf_min``, on
distinct chromosomes when possible) share ``large_frac`` of the genetic
variance; the other ``n_causal - n_large`` causal variants are polygenic with
standardised effects ~ N(0, 1). Covariates: age ~ U(40, 70) and sex ~
Bernoulli(0.5) with modest effects. Height is finally standardised (training
mean 0, sd 1); truth betas are reported on that final scale.

A held-out test cohort (``n_test`` samples from the same population and the
same effects) is written to ``test/`` for out-of-sample R2 / PRS checks.

The run is deterministic for a given seed and independent of ``n_workers``.
Presets: ``"n20k"`` (n = 20,000 + 5,000 held out, 1M variants, ~12 GB; the
defaults) and ``"n3k"`` (n = 3,000 + 1,000 held out, 500k variants, ~1 GB).
numpy's random distribution streams are not guaranteed stable across numpy
versions, so ``meta.json`` records the numpy version and a sha256 per file;
``cugen.datasets.verify`` compares a regenerated copy against them.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path

import numpy as np

from .._codec import BED_MAGIC, pack_bed
from ..write import CugenWriter

logger = logging.getLogger("cugen.datasets")

# GRCh37 autosome lengths (bp); only their ratios are used.
GRCH37_AUTOSOME_BP = (
    249250621, 243199373, 198022430, 191154276, 180915260, 171115067,
    159138663, 146364022, 141213431, 135534747, 135006516, 133851895,
    115169878, 107349540, 102531392, 90354753, 81195210, 78077248,
    59128983, 63025520, 48129895, 51304566)

_BASES = np.array(list("ACGT"))
STANDARD_COVARIATES = ["age", "sex"] + [f"PC{i}" for i in range(1, 11)]


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------
DEFAULT_PARAMS = dict(
    n=20_000, m=1_000_000, n_chrom=22, h2=0.8, n_causal=2_000, n_large=10, large_frac=0.10,
    n_pops=1, fst=0.0, strat_shift=0.0, block_mean=150, n_founders=32, maf_min=0.01, seed=2026,
    formats=("cugen", "bed"), n_test=5_000, maf_beta=(1.0, 1.5), spacing_bp=2_900,
    ld_scale_bp=100_000, switch_rate=0.01, founder_conc=0.5, large_maf_min=0.05,
    age_effect=-0.01, sex_effect=0.4, n_pcs=10, pc_max_snps=5_000, chrom_lengths=None)

# Named parameter sets; explicit keyword arguments override them.
PRESETS = {
    "n20k": {},                                              # the defaults (~12 GB, ~7 min on 8 CPUs)
    "n3k": dict(n=3_000, n_test=1_000, m=500_000),           # small demo (~1 GB)
}


def _log(verbose, msg):
    if verbose:
        logger.info("[simulate_height] %s", msg)


def _chrom_lengths(spec, n_chrom):
    """Relative chromosome lengths: "grch37" (GRCh37 autosome ratios, n_chrom <= 22),
    "equal", or an explicit sequence of n_chrom lengths. None = "grch37" up to 22
    chromosomes, else "equal"."""
    if spec is None:
        spec = "grch37" if n_chrom <= len(GRCH37_AUTOSOME_BP) else "equal"
    if isinstance(spec, str):
        if spec == "grch37":
            if not 1 <= n_chrom <= len(GRCH37_AUTOSOME_BP):
                raise ValueError(f"chrom_lengths='grch37' needs n_chrom in 1..{len(GRCH37_AUTOSOME_BP)}; "
                                 f"use chrom_lengths='equal' for more chromosomes")
            return "grch37", tuple(GRCH37_AUTOSOME_BP[:n_chrom])
        if spec == "equal":
            if n_chrom < 1:
                raise ValueError("n_chrom must be >= 1")
            return "equal", (1,) * n_chrom
        raise ValueError("chrom_lengths must be 'grch37', 'equal' or a sequence of lengths")
    lens = tuple(float(x) for x in spec)
    if len(lens) != n_chrom or min(lens) <= 0:
        raise ValueError(f"chrom_lengths needs {n_chrom} positive lengths")
    return [float(x) for x in lens], lens


def _sha256(path, bufsize=8 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(bufsize)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _split_counts(total, weights):
    """Largest-remainder split of ``total`` into len(weights) positive ints."""
    w = np.asarray(weights, dtype=np.float64)
    raw = total * w / w.sum()
    out = np.floor(raw).astype(np.int64)
    rem = int(total - out.sum())
    if rem > 0:
        out[np.argsort(-(raw - out), kind="stable")[:rem]] += 1
    if (out < 1).any():
        raise ValueError(f"m={total} too small for {len(w)} chromosomes")
    return out


def _dirichlet(rng, alpha):
    """Dirichlet via gamma draws, robust to tiny concentration parameters."""
    a = np.maximum(np.asarray(alpha, dtype=np.float64), 1e-6)
    g = rng.standard_gamma(a)
    s = g.sum()
    if not np.isfinite(s) or s <= 0:          # every gamma underflowed
        g = np.zeros_like(a)
        g[np.argmax(a)] = 1.0
        s = 1.0
    return g / s


def _top_pcs(X_train, X_test, k, rng, n_iter=6, oversample=20):
    """Randomised SVD PCs of a standardised (n, B) matrix; test samples projected."""
    n, B = X_train.shape
    k = min(k, B, n)
    if k <= 0:
        return np.zeros((n, 0)), np.zeros((X_test.shape[0], 0))
    q = min(k + oversample, B)
    Om = rng.standard_normal((B, q)).astype(np.float32)
    Y = X_train @ Om
    for _ in range(n_iter):
        Y, _r = np.linalg.qr(Y)
        Z, _r = np.linalg.qr(X_train.T @ Y)
        Y = X_train @ Z
    Qm, _r = np.linalg.qr(Y)
    Bm = Qm.T @ X_train
    _u, _s, Vt = np.linalg.svd(Bm.astype(np.float64), full_matrices=False)
    V = Vt[:k].T                                   # (B, k)
    # deterministic sign: largest |loading| positive
    sgn = np.sign(V[np.abs(V).argmax(axis=0), np.arange(k)])
    sgn[sgn == 0] = 1.0
    V = V * sgn
    sc = 1.0 / math.sqrt(B)
    return (X_train.astype(np.float64) @ V) * sc, (X_test.astype(np.float64) @ V) * sc


# --------------------------------------------------------------------------
# plan: variant table, blocks, causal variants (main process, deterministic)
# --------------------------------------------------------------------------
def _plan(p, rng):
    n_chrom = p["n_chrom"]
    _, chrom_len = _chrom_lengths(p["chrom_lengths"], n_chrom)
    m_c = _split_counts(p["m"], chrom_len)
    chroms = []
    gidx0 = 0
    for ci in range(n_chrom):
        mc = int(m_c[ci])
        # LD blocks: gamma(shape 2) lengths with mean block_mean, >= 2 variants
        sizes = []
        tot = 0
        while tot < mc:
            s = max(2, int(round(rng.gamma(2.0, p["block_mean"] / 2.0))))
            sizes.append(s)
            tot += s
        sizes[-1] -= tot - mc
        if sizes[-1] < 2 and len(sizes) > 1:
            sizes[-2] += sizes[-1]
            sizes.pop()
        starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
        gaps = np.maximum(1, np.round(rng.exponential(p["spacing_bp"], mc))).astype(np.int64)
        pos = 10_000 + np.cumsum(gaps)
        # target MAF ~ truncated 0.5 * Beta(a, b) on [maf_min, 0.5]
        a, b = p["maf_beta"]
        maf = np.empty(mc)
        fill = 0
        while fill < mc:
            d = 0.5 * rng.beta(a, b, size=2 * (mc - fill) + 16)
            d = d[d >= p["maf_min"]][: mc - fill]
            maf[fill:fill + d.size] = d
            fill += d.size
        alt_minor = rng.random(mc) < 0.5
        af = np.where(alt_minor, maf, 1.0 - maf)
        pair = rng.permuted(np.tile(np.arange(4), (mc, 1)), axis=1)[:, :2]
        ref, alt = _BASES[pair[:, 0]], _BASES[pair[:, 1]]
        chroms.append(dict(chrom=ci + 1, m=mc, gidx0=gidx0, pos=pos, af=af,
                           ref=ref, alt=alt, block_starts=starts,
                           block_sizes=np.asarray(sizes, dtype=np.int64)))
        gidx0 += mc

    # ---- causal variants ------------------------------------------------
    m = p["m"]
    n_large, n_causal = p["n_large"], p["n_causal"]
    if n_large > n_causal:
        raise ValueError("n_large must be <= n_causal")
    chrom_of = np.concatenate([np.full(c["m"], c["chrom"]) for c in chroms])
    pos_all = np.concatenate([c["pos"] for c in chroms])
    af_all = np.concatenate([c["af"] for c in chroms])
    block_all = np.concatenate([
        np.repeat(np.arange(len(c["block_sizes"])), c["block_sizes"]) for c in chroms])
    block_uid = chrom_of.astype(np.int64) * 10_000_000 + block_all
    maf_all = np.minimum(af_all, 1 - af_all)

    large = []
    if n_large > 0:
        lens = np.asarray(chrom_len, dtype=np.float64)
        if n_large <= n_chrom:
            cs = rng.choice(np.arange(1, n_chrom + 1), size=n_large, replace=False, p=lens / lens.sum())
        else:
            cs = rng.choice(np.arange(1, n_chrom + 1), size=n_large, replace=True, p=lens / lens.sum())
        for c in cs:
            ok = np.flatnonzero((chrom_of == c) & (maf_all >= p["large_maf_min"]))
            if ok.size == 0:
                ok = np.flatnonzero(chrom_of == c)
            # keep away from chromosome ends and from earlier large loci (5 Mb)
            lo, hi = pos_all[ok].min(), pos_all[ok].max()
            span = hi - lo
            inner = ok[(pos_all[ok] > lo + 0.05 * span) & (pos_all[ok] < hi - 0.05 * span)]
            cand = inner if inner.size else ok
            for _try in range(200):
                j = int(rng.choice(cand))
                if all(chrom_of[j] != chrom_of[k] or abs(pos_all[j] - pos_all[k]) > 5_000_000
                       for k in large):
                    break
            large.append(j)
    large = np.asarray(large, dtype=np.int64)
    excl = np.isin(block_uid, block_uid[large]) if large.size else np.zeros(m, bool)
    pool = np.flatnonzero(~excl)
    n_poly = n_causal - n_large
    if n_poly > pool.size:
        raise ValueError("n_causal too large for m")
    poly = np.sort(rng.choice(pool, size=n_poly, replace=False)) if n_poly else np.zeros(0, np.int64)
    causal = np.concatenate([large, poly]).astype(np.int64)
    is_large = np.concatenate([np.ones(large.size, bool), np.zeros(poly.size, bool)])
    lf = p["large_frac"] if (n_large and n_poly) else (1.0 if n_large else 0.0)
    share = np.zeros(causal.size)
    if n_large:
        share[:n_large] = lf * _dirichlet(rng, np.full(n_large, 20.0))
    if n_poly:
        z2 = rng.standard_normal(n_poly) ** 2
        share[n_large:] = (1.0 - lf) * z2 / z2.sum()
    sign = np.where(rng.random(causal.size) < 0.5, -1.0, 1.0)
    order = np.argsort(causal, kind="stable")
    causal, is_large, share, sign = causal[order], is_large[order], share[order], sign[order]
    return chroms, dict(gidx=causal, is_large=is_large, share=share, sign=sign,
                        block_uid=block_uid, chrom_of=chrom_of)


# --------------------------------------------------------------------------
# per-chromosome worker
# --------------------------------------------------------------------------
def _sim_chrom(task):
    t0 = time.time()
    c = task["chrom"]
    p = task["params"]
    rng = np.random.default_rng(task["seed"])
    n, n_test = p["n"], p["n_test"]
    nt = n + n_test
    H, Htr = 2 * nt, 2 * n
    K = p["n_founders"]
    P = p["n_pops"]
    fst = p["fst"]
    pop = task["pop"]
    hpop = np.repeat(pop, 2)
    pop_idx = [np.flatnonzero(hpop == q) for q in range(P)]
    pos, af_t = task["pos"], task["af"]
    gidx0 = task["gidx0"]
    causal_local = task["causal_local"]        # sorted local indices
    causal_coef = task["causal_coef"]          # sign * sqrt(share)
    out = task["out"]
    fmt = p["formats"]

    g_raw = np.zeros(nt, dtype=np.float64)
    af_real = np.zeros(task["m"], dtype=np.float64)
    causal_rec = []
    pc_cols = []
    w_tr = w_te = None
    f_tr = f_te = None
    if "cugen" in fmt:
        w_tr = CugenWriter(out["cugen"], n, task["m"]).__enter__()
        if n_test:
            w_te = CugenWriter(out["cugen_test"], n_test, task["m"]).__enter__()
    if "bed" in fmt:
        f_tr = open(out["bed_part"], "wb")
        if n_test:
            f_te = open(out["bed_part_test"], "wb")
    ci = 0
    try:
        for bs, bl in zip(task["block_starts"], task["block_sizes"]):
            L = int(bl)
            sl = slice(bs, bs + L)
            bpos = pos[sl]
            gaps = np.diff(bpos).astype(np.float64)
            # founder frequencies (Balding-Nichols at the haplotype level)
            w = _dirichlet(rng, np.full(K, p["founder_conc"]))
            if P > 1 and fst > 0:
                Wp = np.stack([_dirichlet(rng, w * (1.0 - fst) / fst) for _ in range(P)])
            else:
                Wp = np.tile(w, (P, 1))
            cdf = np.cumsum(Wp, axis=1)
            cdf[:, -1] = 1.0

            def draw(hidx):
                u = rng.random(hidx.size)
                f = np.empty(hidx.size, dtype=np.int8)
                hp = hpop[hidx]
                for q in range(P):
                    m_ = hp == q
                    f[m_] = np.minimum(np.searchsorted(cdf[q], u[m_], side="right"), K - 1)
                return f

            # copying path z (L, H): founder index per position and haplotype
            src = np.zeros((L, H), dtype=np.int8)
            src[0] = draw(np.arange(H))
            tix = np.zeros((L, H), dtype=np.int16 if L < 32000 else np.int32)
            if L > 1:
                psw = (1.0 - np.exp(-p["switch_rate"] * gaps / p["spacing_bp"])).astype(np.float32)
                sw = rng.random((L - 1, H), dtype=np.float32) < psw[:, None]
                rr, hh = np.nonzero(sw)
                src[rr + 1, hh] = draw(hh)
                tix[rr + 1, hh] = (rr + 1).astype(tix.dtype)
                np.maximum.accumulate(tix, axis=0, out=tix)
                z = np.take_along_axis(src, tix.astype(np.intp), axis=0)
                del sw, rr, hh
            else:
                z = src
            del src, tix
            # founder shares per position in the TRAINING haplotypes
            flat = z[:, :Htr].astype(np.int32) + (K * np.arange(L, dtype=np.int32))[:, None]
            shares = np.bincount(flat.ravel(), minlength=L * K).reshape(L, K) / float(Htr)
            del flat
            # latent AR(1) founder process along the block
            u = np.empty((L, K))
            u[0] = rng.standard_normal(K)
            if L > 1:
                rho = np.exp(-gaps / p["ld_scale_bp"])
                eps = rng.standard_normal((L - 1, K))
                for j in range(1, L):
                    u[j] = rho[j - 1] * u[j - 1] + math.sqrt(1.0 - rho[j - 1] ** 2) * eps[j - 1]
            order = np.argsort(u, axis=1, kind="stable")
            rank = np.argsort(order, axis=1, kind="stable").astype(np.int16)
            s_sorted = np.take_along_axis(shares, order, axis=1)
            cs = np.cumsum(s_sorted, axis=1)
            v = rng.random(H)                      # nested key for partial founders
            zr = (z.astype(np.int32) + (K * np.arange(L, dtype=np.int32))[:, None])
            rz = rank.ravel()[zr]                  # (L, H) rank of the copied founder
            del zr, z
            target = af_t[sl].copy()
            G = np.empty((L, nt), dtype=np.int8)
            a_tr = np.zeros(L)
            rows = np.arange(L)
            for _it in range(8):
                tt, css, sss = target[rows], cs[rows], s_sorted[rows]
                t = np.minimum((css <= tt[:, None] + 1e-12).sum(axis=1), K)   # founders fully ALT
                prev = np.where(t > 0, np.take_along_axis(css, np.maximum(t - 1, 0)[:, None], 1)[:, 0], 0.0)
                sp = np.take_along_axis(sss, np.minimum(t, K - 1)[:, None], 1)[:, 0]
                frac = np.where((t < K) & (sp > 0), (tt - prev) / np.maximum(sp, 1e-300), 0.0)
                frac = np.clip(frac, 0.0, 1.0)                                  # partial founder
                rzr = rz if _it == 0 else rz[rows]
                hap = (rzr < t[:, None]) | ((rzr == t[:, None]) & (v[None, :] < frac[:, None]))
                G[rows] = hap[:, 0::2].astype(np.int8) + hap[:, 1::2]
                a_tr[rows] = G[rows, :n].mean(axis=1, dtype=np.float64) / 2.0
                a_r = a_tr[rows]
                bad = np.minimum(a_r, 1.0 - a_r) < p["maf_min"]
                if not bad.any():
                    break
                rows = rows[bad]
                bump = p["maf_min"] * (1.15 + 0.25 * _it)
                target[rows] = np.where(target[rows] < 0.5, bump, 1.0 - bump)
            del hap, rz, rzr
            maf_tr = np.minimum(a_tr, 1.0 - a_tr)
            af_real[sl] = a_tr

            # causal variants in this block -> genetic value
            while ci < causal_local.size and causal_local[ci] < bs + L:
                j = int(causal_local[ci]) - bs
                x = G[j].astype(np.float64)
                sd = float(x[:n].std())
                br = causal_coef[ci] / sd if sd > 0 else 0.0
                g_raw += br * x
                causal_rec.append((gidx0 + bs + j, br, float(a_tr[j]), sd))
                ci += 1
            # one PC-panel variant per block: the most common one
            j_pc = int(np.argmax(maf_tr))
            if maf_tr[j_pc] >= 0.05:
                pc_cols.append(G[j_pc].copy())

            # write
            if w_tr is not None:
                for j in range(L):
                    w_tr.add_variant(gidx0 + bs + j, G[j, :n])
                    if w_te is not None:
                        w_te.add_variant(gidx0 + bs + j, G[j, n:])
            if f_tr is not None:
                f_tr.write(pack_bed(G[:, :n], counted="A1").tobytes())
                if f_te is not None:
                    f_te.write(pack_bed(G[:, n:], counted="A1").tobytes())
            del G
    finally:
        for wr in (w_tr, w_te):
            if wr is not None:
                wr.__exit__(None, None, None)
        for fh in (f_tr, f_te):
            if fh is not None:
                fh.close()
    pc = np.stack(pc_cols) if pc_cols else np.zeros((0, nt), dtype=np.int8)
    return dict(chrom=c, g_raw=g_raw, af=af_real, causal=causal_rec, pc=pc,
                seconds=time.time() - t0)


# --------------------------------------------------------------------------
# public entry point
# --------------------------------------------------------------------------
def simulate_height(out_dir, preset=None, *, n_workers=1, verbose=True, checksums=True, **params):
    """Simulate a synthetic "exaggerated human height" GWAS dataset into ``out_dir``.

    ``preset`` names a parameter set (``"n20k"`` = the defaults, ``"n3k"`` = a small
    demo; see :data:`PRESETS`); keyword arguments override it.

    Parameters (see the module docstring for the model)
    ----------
    out_dir : output directory (created).
    preset : None (the defaults below) | "n20k" | "n3k".
    n : training samples (the GWAS cohort; written to ``cugen/`` and ``bed/``). Default 20,000.
    m : total variants over ``n_chrom`` autosomes. Default 1,000,000.
    n_chrom : number of autosomes. Default 22.
    chrom_lengths : "grch37" (GRCh37 length ratios; needs n_chrom <= 22) | "equal" |
        a sequence of n_chrom relative lengths. None = "grch37" up to 22 chromosomes,
        else "equal".
    h2 : SNP heritability of the covariate-adjusted trait (exact in training). Default 0.8.
    n_causal, n_large, large_frac : causal variants; ``n_large`` of them share
        ``large_frac`` of the genetic variance. Defaults 2,000 / 10 / 0.10.
    n_pops, fst, strat_shift : population structure (Balding-Nichols at the
        haplotype level) and a population mean shift of the phenotype.
    block_mean, n_founders : mean LD-block length (variants) and founders/block.
    maf_min : minimum MAF in the training samples. Default 0.01.
    seed : master seed (default 2026); the output is identical for a seed whatever
        ``n_workers``.
    formats : any of "cugen", "bed".
    n_test : held-out samples written to ``test/`` (0 = none). Default 5,000.
    maf_beta : target MAF ~ 0.5 * Beta(a, b) truncated at maf_min.
    spacing_bp, ld_scale_bp, switch_rate, founder_conc : LD-shape controls.
    large_maf_min : minimum MAF of a large-effect locus.
    age_effect, sex_effect : covariate effects (per year from 55; male vs female)
        on the covariate-adjusted scale (total genetic+noise variance 1).
    n_pcs, pc_max_snps : PCs written as PC1..PCk, computed from one common
        variant per LD block (at most pc_max_snps), test samples projected.
    n_workers : processes over chromosomes (does not change the output).
    verbose : log progress on the ``cugen.datasets`` logger.
    checksums : record the size and sha256 of every file in ``out_dir`` in ``meta.json``
        (``cugen.datasets.verify`` checks them); use a fresh ``out_dir``.

    Every parameter is recorded in ``meta.json``. The UltraLasso workflow of this
    release analyses chromosomes 1..22; datasets with more chromosomes are for the
    background methods (``cugen.background``), which take any chromosome set.

    Returns
    -------
    dict : the meta record also written to ``meta.json``.
    """
    from .. import __version__ as _cg_version
    t_start = time.time()
    if preset is not None and preset not in PRESETS:
        raise ValueError(f"unknown preset {preset!r}; choose one of {sorted(PRESETS)}")
    unknown = set(params) - set(DEFAULT_PARAMS)
    if unknown:
        raise TypeError(f"simulate_height() got unknown parameters {sorted(unknown)}")
    kw = dict(DEFAULT_PARAMS)
    if preset is not None:
        kw.update(PRESETS[preset])
    kw.update(params)
    out = Path(out_dir)
    formats = tuple(kw["formats"])
    bad = set(formats) - {"cugen", "bed"}
    if bad:
        raise ValueError(f"unknown formats {sorted(bad)}")
    if not 0 < kw["h2"] < 1:
        raise ValueError("h2 must be in (0, 1)")
    if kw["n_pops"] > 1 and not 0 < kw["fst"] < 1:
        raise ValueError("n_pops > 1 needs 0 < fst < 1")
    if kw["n_founders"] > 127:
        raise ValueError("n_founders must be <= 127")
    chrom_lengths, _ = _chrom_lengths(kw["chrom_lengths"], int(kw["n_chrom"]))
    maf_beta = kw["maf_beta"]
    seed = int(kw["seed"])
    params = dict(n=int(kw["n"]), m=int(kw["m"]), n_chrom=int(kw["n_chrom"]), h2=float(kw["h2"]),
                  n_causal=int(kw["n_causal"]), n_large=int(kw["n_large"]),
                  large_frac=float(kw["large_frac"]), n_pops=int(kw["n_pops"]), fst=float(kw["fst"]),
                  strat_shift=float(kw["strat_shift"]), block_mean=float(kw["block_mean"]),
                  n_founders=int(kw["n_founders"]), maf_min=float(kw["maf_min"]), seed=seed,
                  formats=list(formats), n_test=int(kw["n_test"]),
                  maf_beta=[float(maf_beta[0]), float(maf_beta[1])],
                  spacing_bp=float(kw["spacing_bp"]), ld_scale_bp=float(kw["ld_scale_bp"]),
                  switch_rate=float(kw["switch_rate"]), founder_conc=float(kw["founder_conc"]),
                  large_maf_min=float(kw["large_maf_min"]), age_effect=float(kw["age_effect"]),
                  sex_effect=float(kw["sex_effect"]), n_pcs=int(kw["n_pcs"]),
                  pc_max_snps=int(kw["pc_max_snps"]), chrom_lengths=chrom_lengths)
    n, n_test = params["n"], params["n_test"]
    nt = n + n_test
    for d in ("cugen", "bed", "test/cugen", "test/bed"):
        if ("test" in d and n_test == 0) or (d.endswith("cugen") and "cugen" not in formats) \
                or (d.endswith("bed") and "bed" not in formats):
            continue
        (out / d).mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    ss = np.random.SeedSequence(params["seed"])
    ss_plan, ss_pop, ss_pheno, ss_pc, ss_chr = ss.spawn(5)
    chr_seeds = ss_chr.spawn(params["n_chrom"])

    _log(verbose, f"plan: n={n:,} (+{n_test:,} test) m={params['m']:,} chrom={params['n_chrom']} seed={seed}")
    chroms, causal = _plan(params, np.random.default_rng(ss_plan))
    rng_pop = np.random.default_rng(ss_pop)
    P = params["n_pops"]
    pop = np.concatenate([np.arange(n) % P, np.arange(n_test) % P]).astype(np.int8)
    pop = np.concatenate([rng_pop.permutation(pop[:n]), rng_pop.permutation(pop[n:])]).astype(np.int8)

    # ---- per-chromosome tasks --------------------------------------------
    tasks = []
    cg = causal["gidx"]
    coef = causal["sign"] * np.sqrt(causal["share"])
    for c in chroms:
        sel = (cg >= c["gidx0"]) & (cg < c["gidx0"] + c["m"])
        tasks.append(dict(
            chrom=c["chrom"], m=c["m"], gidx0=c["gidx0"], pos=c["pos"], af=c["af"],
            block_starts=c["block_starts"], block_sizes=c["block_sizes"],
            causal_local=(cg[sel] - c["gidx0"]).astype(np.int64), causal_coef=coef[sel],
            seed=chr_seeds[c["chrom"] - 1], pop=pop, params=params,
            out=dict(cugen=str(out / "cugen" / f"chr{c['chrom']}.cugen"),
                     cugen_test=str(out / "test" / "cugen" / f"chr{c['chrom']}.cugen"),
                     bed_part=str(out / "bed" / f".chr{c['chrom']}.bedpart"),
                     bed_part_test=str(out / "test" / "bed" / f".chr{c['chrom']}.bedpart"))))
    t_gen = time.time()
    results = {}
    if n_workers and n_workers > 1:
        import multiprocessing as mp
        ctx = mp.get_context("fork") if os.name == "posix" else mp.get_context("spawn")
        order = sorted(range(len(tasks)), key=lambda i: -tasks[i]["m"])   # LPT
        with ctx.Pool(min(n_workers, len(tasks))) as pool:
            for r in pool.imap_unordered(_sim_chrom, [tasks[i] for i in order]):
                results[r["chrom"]] = r
                _log(verbose, f"chr{r['chrom']} done in {r['seconds']:.1f}s")
    else:
        for tk in tasks:
            r = _sim_chrom(tk)
            results[r["chrom"]] = r
            _log(verbose, f"chr{r['chrom']} done in {r['seconds']:.1f}s")
    t_gen = time.time() - t_gen
    res = [results[c["chrom"]] for c in chroms]

    # ---- PLINK files: concatenate bed bodies, bim, fam --------------------
    fids = np.array([str(1_000_001 + i) for i in range(n)])
    fids_test = np.array([str(2_000_001 + i) for i in range(n_test)])
    rng_ph = np.random.default_rng(ss_pheno)
    age = np.round(rng_ph.uniform(40, 70, nt), 1)
    sex = (rng_ph.random(nt) < 0.5).astype(np.int8)        # 1 = male, 0 = female
    chr_col = np.concatenate([np.full(c["m"], c["chrom"]) for c in chroms])
    pos_col = np.concatenate([c["pos"] for c in chroms])
    ref_col = np.concatenate([c["ref"] for c in chroms])
    alt_col = np.concatenate([c["alt"] for c in chroms])
    ids = np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(np.char.add(
        chr_col.astype(str), ":"), pos_col.astype(str)), ":"), ref_col), ":"), alt_col)
    if "bed" in formats:
        for sub, stem, ff, ss_ in (("bed", "height", fids, slice(0, n)),
                                   ("test/bed", "height_test", fids_test, slice(n, nt))):
            if ff.size == 0:
                continue
            with open(out / sub / f"{stem}.bed", "wb") as fo:
                fo.write(BED_MAGIC)
                for c in chroms:
                    part = out / sub / f".chr{c['chrom']}.bedpart"
                    with open(part, "rb") as fi:
                        shutil.copyfileobj(fi, fo, 16 << 20)
                    part.unlink()
            with open(out / sub / f"{stem}.bim", "w") as fo:
                cm = pos_col / 1e6
                for k in range(0, len(ids), 200_000):
                    s = slice(k, k + 200_000)
                    fo.write("".join(f"{a}\t{b}\t{c_:.6f}\t{d}\t{e}\t{f}\n" for a, b, c_, d, e, f in
                                     zip(chr_col[s], ids[s], cm[s], pos_col[s], alt_col[s], ref_col[s])))
            sx = sex[ss_]
            with open(out / sub / f"{stem}.fam", "w") as fo:
                fo.write("".join(f"{f}\t{f}\t0\t0\t{1 if s_ == 1 else 2}\t-9\n" for f, s_ in zip(ff, sx)))

    # ---- phenotype -------------------------------------------------------
    g = np.sum([r["g_raw"] for r in res], axis=0)
    g = g - g[:n].mean()
    vg = float(g[:n].var())
    scale = math.sqrt(params["h2"] / vg) if vg > 0 else 0.0
    g *= scale
    e = rng_ph.normal(0.0, math.sqrt(1.0 - params["h2"]), nt)
    cov = params["age_effect"] * (age - 55.0) + params["sex_effect"] * (sex - 0.5)
    popc = (pop - (P - 1) / 2.0) / max((P - 1) / 2.0, 1.0) if P > 1 else np.zeros(nt)
    pshift = params["strat_shift"] * popc
    y = g + e + cov + pshift
    mu_y, sd_y = float(y[:n].mean()), float(y[:n].std())
    height = (y - mu_y) / sd_y
    h2_real = float(g[:n].var() / (g[:n].var() + e[:n].var()))

    # ---- PCs ----------------------------------------------------------------
    pcg = np.concatenate([r["pc"] for r in res], axis=0) if res else np.zeros((0, nt), np.int8)
    rng_pc = np.random.default_rng(ss_pc)
    if pcg.shape[0] > params["pc_max_snps"]:
        keep = np.sort(rng_pc.choice(pcg.shape[0], params["pc_max_snps"], replace=False))
        pcg = pcg[keep]
    Xp = pcg.T.astype(np.float32)                    # (nt, B)
    mu = Xp[:n].mean(axis=0)
    sdp = Xp[:n].std(axis=0)
    sdp[sdp == 0] = 1.0
    Xp = (Xp - mu) / sdp
    pcs_tr, pcs_te = _top_pcs(Xp[:n], Xp[n:], params["n_pcs"], rng_pc)
    del Xp, pcg
    pcs = np.concatenate([pcs_tr, pcs_te], axis=0)
    n_pc_snps = int(min(params["pc_max_snps"], sum(r["pc"].shape[0] for r in res)))

    # ---- tables ----------------------------------------------------------
    import pandas as pd
    pc_names = [f"PC{i}" for i in range(1, pcs.shape[1] + 1)]

    def _pheno_frame(sl, ff):
        d = {"FID": ff, "IID": ff, "height": np.round(height[sl], 6), "age": age[sl],
             "sex": sex[sl]}
        for i, nm in enumerate(pc_names):
            d[nm] = np.round(pcs[sl, i], 6)
        if P > 1:
            d["pop"] = pop[sl]
        return pd.DataFrame(d)

    covars = ["age", "sex"] + pc_names
    for sub, sl, ff in (("", slice(0, n), fids), ("test", slice(n, nt), fids_test)):
        if ff.size == 0:
            continue
        dd = out / sub
        dd.mkdir(parents=True, exist_ok=True)
        ph = _pheno_frame(sl, ff)
        ph.to_csv(dd / "pheno.tsv", sep="\t", index=False)
        ph[["FID", "IID"] + covars].to_csv(dd / "covar.tsv", sep="\t", index=False)
        pd.DataFrame({"#FID": ff, "IID": ff, "SEX": np.where(sex[sl] == 1, 1, 2)}).to_csv(
            dd / "samples.psam", sep="\t", index=False)
        np.savez(dd / "cohort_ref.npz", sample_idx_sorted=np.arange(ff.size, dtype=np.int64),
                 fid_order=ff.astype(str))
        if (dd / "cugen").exists():
            for c in chroms:     # <cugen>.samples.txt sidecars (cugen convention)
                with open(dd / "cugen" / f"chr{c['chrom']}.cugen.samples.txt", "w") as fo:
                    fo.write("\n".join(ff) + "\n")

    annot = pd.DataFrame({"gidx": np.arange(params["m"], dtype=np.int64),
                          "CHR": chr_col.astype(np.int64), "POS": pos_col.astype(np.int64),
                          "ID": ids, "REF": ref_col, "ALT": alt_col})
    annot.to_feather(out / "annotation.feather")
    if n_test:
        shutil.copyfile(out / "annotation.feather", out / "test" / "annotation.feather")

    # ld blocks
    brow = []
    for c in chroms:
        for b, (bs, bl) in enumerate(zip(c["block_starts"], c["block_sizes"])):
            brow.append((c["chrom"] * 10_000_000 + b, c["chrom"], c["gidx0"] + bs,
                         c["gidx0"] + bs + bl - 1, int(c["pos"][bs]), int(c["pos"][bs + bl - 1]), int(bl)))
    blocks = pd.DataFrame(brow, columns=["block", "CHR", "gidx_start", "gidx_end", "start", "end", "n_variants"])
    blocks["n_causal"] = blocks["block"].map(pd.Series(causal["block_uid"][cg]).value_counts()).fillna(0).astype(int)
    blocks.to_csv(out / "ld_blocks.tsv", sep="\t", index=False)

    # truth
    rec = {}
    for r in res:
        for gi, br, a_, sd in r["causal"]:
            rec[gi] = (br, a_, sd)
    br = np.array([rec[i][0] for i in cg])
    af_c = np.array([rec[i][1] for i in cg])
    sd_c = np.array([rec[i][2] for i in cg])
    beta_adj = scale * br                                 # per ALT allele, adjusted scale
    truth = pd.DataFrame({
        "gidx": cg, "ID": ids[cg], "CHR": chr_col[cg], "POS": pos_col[cg],
        "REF": ref_col[cg], "ALT": alt_col[cg], "AF": np.round(af_c, 6),
        "beta": beta_adj / sd_y, "beta_std": beta_adj * sd_c,
        "var_explained": (beta_adj * sd_c) ** 2, "is_large": causal["is_large"].astype(int),
        "block": causal["block_uid"][cg]})
    truth.to_csv(out / "truth.tsv", sep="\t", index=False, float_format="%.6g")
    np.savez_compressed(out / "genetic_value.npz", fid=np.concatenate([fids, fids_test]),
                        g=(g / sd_y).astype(np.float32), split=np.r_[np.zeros(n, np.int8), np.ones(n_test, np.int8)])

    af_all = np.concatenate([r["af"] for r in res]).astype(np.float64)
    maf_all = np.minimum(af_all, 1 - af_all)
    hist_edges = [0.0, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
    hist = np.histogram(maf_all, bins=hist_edges)[0].tolist()

    # config for the standard workflow (paths relative to the dataset dir); the UltraLasso
    # workflow of this release analyses chromosomes 1..22 only
    from ..config import DEFAULTS as _CFG
    if params["n_chrom"] <= len(_CFG["data"]["chromosomes"]):
        write_workflow_config(out / "config_ultralasso.json", data_dir=".", output_dir="results/ultralasso",
                              n=n, covariates=covars, relative=True,
                              chromosomes=range(1, params["n_chrom"] + 1))
    else:
        _log(verbose, f"no config_ultralasso.json: {params['n_chrom']} chromosomes is beyond the "
                      f"workflow's chromosome set")

    sizes = {}
    for sub in ("cugen", "bed", "test"):
        dd = out / sub
        if dd.exists():
            sizes[sub] = int(sum(f.stat().st_size for f in dd.rglob("*") if f.is_file()))
    meta = dict(
        generator="cugen.datasets.simulate_height", cugen_version=_cg_version,
        build="synthetic", params=params, n_variants_per_chrom={str(c["chrom"]): c["m"] for c in chroms},
        n_blocks=int(len(blocks)), n_blocks_with_causal=int((blocks["n_causal"] > 0).sum()),
        realised=dict(h2=h2_real, var_g=float(g[:n].var()), var_e=float(e[:n].var()),
                      var_cov=float(cov[:n].var()), var_popshift=float(pshift[:n].var()),
                      sd_y_before_standardisation=sd_y, mean_y_before_standardisation=mu_y,
                      h2_observed_scale=float(g[:n].var() / y[:n].var()),
                      large_var_explained=float(truth.loc[truth.is_large == 1, "var_explained"].sum()),
                      poly_var_explained=float(truth.loc[truth.is_large == 0, "var_explained"].sum()),
                      maf_min=float(maf_all.min()), maf_mean=float(maf_all.mean()),
                      maf_hist_edges=hist_edges, maf_hist=hist,
                      n_pc_snps=n_pc_snps),
        allele_coding=dict(cugen="ALT dosage (0/1/2)", bed="bim column 5 (A1) = ALT, column 6 = REF; "
                           "plink/plink2/REGENIE count A1 = ALT",
                           annotation="REF/ALT as in the cugen ALT dosage",
                           counted_allele="ALT"),
        h2_definition="SNP heritability of the covariate-adjusted trait (g + e), exact in training samples",
        timings_s=dict(generate=round(t_gen, 1), total=round(time.time() - t_start, 1),
                       per_chrom={str(r["chrom"]): round(r["seconds"], 1) for r in res}),
        sizes_bytes=sizes, n_workers=int(n_workers or 1),
        numpy_version=np.__version__, preset=preset,
        regenerate=regenerate_command(params, preset=preset, out="DIR"))
    _write_readme(out, meta)
    if checksums:
        # every file in out_dir except meta.json itself (README.md is written first, so it is
        # covered); generate into a fresh directory to keep the list to the dataset's own files
        t_sha = time.time()
        meta["files"] = {f.relative_to(out).as_posix(): dict(size=int(f.stat().st_size), sha256=_sha256(f))
                         for f in sorted(out.rglob("*"))
                         if f.is_file() and f.relative_to(out).as_posix() != "meta.json"}
        meta["timings_s"]["checksums"] = round(time.time() - t_sha, 1)
    with open(out / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    _log(verbose, f"done in {time.time() - t_start:.1f}s (generate {t_gen:.1f}s); h2_realised={h2_real:.4f}")
    return meta


def regenerate_command(params=None, preset=None, out="DIR"):
    """The ``python -m cugen.datasets.simulate_height`` command that regenerates a dataset
    (the preset, plus every parameter that differs from it)."""
    base = dict(DEFAULT_PARAMS)
    if preset is not None:
        base.update(PRESETS[preset])
    cmd = ["python -m cugen.datasets.simulate_height", f"--out {out}"]
    if preset is not None:
        cmd.append(f"--preset {preset}")
    for k, v in (params or {}).items():
        if k not in base:
            continue
        b = base[k]
        if k == "formats":
            if list(v) != list(b):
                cmd.append(f"--formats {','.join(v)}")
            continue
        if k == "maf_beta":
            if [float(x) for x in v] != [float(x) for x in b]:
                cmd.append(f"--maf-beta {v[0]},{v[1]}")
            continue
        if k == "chrom_lengths":
            auto = _chrom_lengths(None, int(params.get("n_chrom", base["n_chrom"])))[0]
            if v != auto:
                cmd.append(f"--chrom-lengths {v if isinstance(v, str) else ','.join(map(str, v))}")
            continue
        if b is None or float(v) != float(b):
            cmd.append(f"--{k.replace('_', '-')} {v}")
    return " ".join(cmd)


# --------------------------------------------------------------------------
# workflow config
# --------------------------------------------------------------------------
REF_N_PENALTY = 400_000
BACKGROUNDS = ("ultralasso",)
ALPHA_SCALINGS = ("none", "sqrt_n")


def workflow_config(data_dir, output_dir, background="ultralasso", alpha_scaling="none", *,
                    n=None, covariates=None, relative=False, name="height", finemapping=False,
                    chromosomes=None, n_workers_gwas=None):
    """Build an ``ultra_workflow`` config dict for a simulated dataset.

    background : "ultralasso" (the standard pipeline). The workflow selector for the other
        backgrounds (CuGenie, CuGenie-K, UltraLOCO) is not part of this release; call
        ``cugen.background`` directly for those.
    alpha_scaling : "none" (default; the package alphas, aimed at biobank-scale n) or "sqrt_n"
        (step-3 alpha x sqrt(400,000 / n)). "sqrt_n" needs core support for
        ``step3_global_lasso.alpha_scaling``; when the installed cugen lacks it this raises
        instead of writing a key the workflow would silently ignore.
    n_workers_gwas : step-4 / fine-mapping processes. None = 1 for n <= 50,000 (single
        process, no CUDA MPS needed; at small n the 8-worker spawn pool is slower and
        needs MPS), else the package default. Scripts that call ``ultra_workflow`` with
        more than one worker need an ``if __name__ == "__main__":`` guard, because the
        spawned workers re-import the main script.

    chromosomes : chromosomes to analyse; None = 1..n_chrom of the dataset (``meta.json``).

    Paths are joined to ``data_dir`` unless ``relative`` (then they stay relative to the
    dataset directory and the workflow must be run from there).
    """
    from ..config import DEFAULTS
    if background not in BACKGROUNDS:
        raise NotImplementedError(
            f"background={background!r}: the workflow runs the UltraLasso background only in this "
            f"release. Fit CuGenie / CuGenie-K / UltraLOCO with cugen.background (e.g. "
            f"cugen.fit_cugenie) and run step 4 on the returned LOCO predictions.")
    if alpha_scaling not in ALPHA_SCALINGS:
        raise ValueError(f"alpha_scaling must be one of {ALPHA_SCALINGS}, got {alpha_scaling!r}")
    if alpha_scaling != "none" and "alpha_scaling" not in DEFAULTS.get("step3_global_lasso", {}):
        raise NotImplementedError(
            f"alpha_scaling={alpha_scaling!r} needs the core config key "
            f"'step3_global_lasso.alpha_scaling', which this version of cugen does not have; "
            f"the workflow would ignore it and run with unscaled alphas. Use alpha_scaling='none', "
            f"or set step3_global_lasso.alpha to the value you want.")
    d = Path(data_dir)
    j = (lambda s: str(s)) if relative else (lambda s: str((d / s).resolve()))
    if n is None:
        n = int(np.load(d / "cohort_ref.npz", allow_pickle=True)["sample_idx_sorted"].size)
    cov = list(covariates) if covariates is not None else list(STANDARD_COVARIATES)
    if chromosomes is None:
        mp_ = d / "meta.json"
        if mp_.exists():
            with open(mp_) as fh:
                chromosomes = range(1, int(json.load(fh)["params"]["n_chrom"]) + 1)
        else:
            chromosomes = DEFAULTS["data"]["chromosomes"]
    chromosomes = sorted(int(c) for c in chromosomes)
    supported = set(int(c) for c in DEFAULTS["data"]["chromosomes"])
    if not set(chromosomes) <= supported:
        raise ValueError(f"chromosomes {sorted(set(chromosomes) - supported)} are outside 1..{max(supported)}, "
                         f"the chromosome set the UltraLasso workflow analyses in this release. Use "
                         f"cugen.background for other chromosome sets.")
    out_dir = str(output_dir) if relative else str(Path(output_dir).resolve())
    cfg = {
        "general": {"name": name, "output_dir": out_dir},
        "data": {"cugen_dir": j("cugen"), "cugen_pattern": "chr{chrom}.cugen",
                 "annotation": j("annotation.feather"), "phe_file": j("pheno.tsv"),
                 "chromosomes": chromosomes},
        "phenotype": {"mode": "single", "name": "height", "type": "continuous",
                      "file": os.path.join(out_dir, "cohort_height.npz")},
        "cohort": {"build": "auto", "phenotype": "height", "ref_npz": j("cohort_ref.npz"),
                   "psam": j("samples.psam"), "n_samples": int(n), "binary": False},
        "covariates": {"columns": cov},
        "step5_finemapping": {"run": True},
        "step5b_finemapping": {"run": bool(finemapping)},
    }
    if n_workers_gwas is None and int(n) <= 50_000:
        n_workers_gwas = 1
    if n_workers_gwas is not None:
        w = int(n_workers_gwas)
        cfg["gpu"] = {"n_workers_gwas": w, "n_workers_finemapping": w, "cuda_mps": w > 1}
        cfg["step5b_finemapping"]["n_workers"] = w
    if alpha_scaling == "sqrt_n":
        cfg["step3_global_lasso"] = {"alpha_scaling": "sqrt_n", "alpha_ref_n": REF_N_PENALTY}
    return cfg


def write_workflow_config(path, data_dir, output_dir, **kw):
    """:func:`workflow_config`, written to ``path`` as JSON."""
    cfg = workflow_config(data_dir, output_dir, **kw)
    with open(path, "w") as f:
        json.dump(cfg, f, indent=2)
    return cfg


def _lengths_text(p):
    cl = p.get("chrom_lengths", "grch37")
    if cl == "grch37":
        return "GRCh37 length ratios"
    if cl == "equal":
        return "equal lengths"
    return "user-given length ratios"


def _write_readme(out, meta):
    p = meta["params"]
    r = meta["realised"]
    txt = f"""# Synthetic "exaggerated human height" dataset (CuGen demo)

Fully synthetic: no real genotypes or phenotypes were used in any way. Free to keep and share.
Generated by `cugen.datasets.simulate_height` (cugen {meta['cugen_version']}), seed {p['seed']};
every parameter is in `meta.json`, with numpy {meta['numpy_version']} and a sha256 per file.
Regenerate with (bit-for-bit under the same numpy version; check with `cugen.datasets.verify`):

    {meta['regenerate']}

## Contents
| path | what |
|---|---|
| `cugen/chr{{1..{p['n_chrom']}}}.cugen` | training genotypes, n = {p['n']:,}, ALT dosage, exact stored stats |
| `bed/height.{{bed,bim,fam}}` | same genotypes as PLINK1 (bim col 5 = A1 = ALT, the allele plink/REGENIE count) |
| `pheno.tsv` | FID IID height age sex PC1..PC{p['n_pcs']}{' pop' if p['n_pops'] > 1 else ''} (height standardised) |
| `covar.tsv` | FID IID age sex PC1..PC{p['n_pcs']} (REGENIE `--covarFile`) |
| `samples.psam`, `cohort_ref.npz` | sample order for `cg.prepare_cohort` (identity order) |
| `annotation.feather` | gidx CHR POS ID REF ALT (build "synthetic") |
| `truth.tsv` | causal variants: gidx ID CHR POS REF ALT AF beta (per ALT, standardised height) beta_std var_explained is_large block |
| `ld_blocks.tsv` | simulated LD blocks (LD is zero between blocks); n_causal per block |
| `genetic_value.npz` | true genetic value g (standardised-height units), train + test |
| `test/` | held-out cohort (n = {p['n_test']:,}): same layout (cugen, bed, pheno, covar) |
| `config_ultralasso.json` | standard `cg.ultra_workflow` config; run it from this directory |

## Model (short)
{p['m']:,} variants over {p['n_chrom']} autosomes ({_lengths_text(p)}, ~{p['spacing_bp']/1000:.1f} kb spacing);
LD blocks (mean {p['block_mean']:.0f} variants) built as mosaics of {p['n_founders']} founder haplotypes
with Markov switching (LD decays inside a block, ~0 between blocks); MAF >= {p['maf_min']}.
Height: SNP h2 = {p['h2']} of the covariate-adjusted trait (realised {r['h2']:.4f});
{p['n_causal']:,} causal variants, of which {p['n_large']} large loci carry {r['large_var_explained']:.3f}
of the adjusted variance; covariates age, sex (modest effects).
Population structure: n_pops = {p['n_pops']}, Fst = {p['fst']}, strat_shift = {p['strat_shift']}.

## Run CuGen (UltraLasso, standard settings)
    cd <this directory>
    python -c "import cugen as cg; cg.ultra_workflow('config_ultralasso.json')"
Then score against the truth:
    python -c "from cugen.datasets import evaluate_gwas as e; \\
               print(e('results/ultralasso/height/sumstats.tsv.gz', '.'))"

## Notes
* `cugen.convert.bed2cugen` counts the bim column-6 allele (A2) by default, so converting
  `bed/height.bed` with it gives REF dosages (2 - ALT). Use the shipped `cugen/` files, which count
  ALT like the annotation; `cugen.BedGenotypes` counts A1 (= ALT here) by default.
* chr6 25-34 Mb is treated as "HLA" by `cg.define_loci` (merged into one locus) even though nothing
  is special about it here.
"""
    (out / "README.md").write_text(txt)
