"""Blocked step 4 on any device: the package's linear offset Wald test, on numpy or cupy.

Same statistic as the linear step 4 of :func:`cugen.gwas` (the ``fused_univariate``
kernel), per chromosome c and LOCO background k:

    r     = y_original - prediction_c,   centred;   syy = r'r
    num   = sum over VALID calls of (g - mu_x) r    (complete case; missing calls contribute 0)
    beta  = num / sxx,    SE = sqrt((syy - beta^2 sxx) / ((n - 2) sxx)),    Z = beta / SE

mu_x / sxx / maf are the cugen's stored per-variant stats (``stats="stored"``, the
package behaviour) or recomputed from the analysed samples' valid calls
(``stats="recompute"``, always used for .bed input). Several LOCO backgrounds are scored
with one decode per block (one GEMM of the block with an n x K residual matrix).

``BETA`` is per copy of the counted allele, which is the ``ALT`` column (for .bed input
ALT := the counted .bim allele and REF := the other one); ``df.attrs["counted_allele"]``
and the log name it. Public entry point: :func:`cugen.background.gwas_blocked`.
"""
from __future__ import annotations

import gzip
import logging
import os
import shutil
import time

import numpy as np

from . import _resources as R
from ._device import device as _device

logger = logging.getLogger("cugen.background.gwas_blocked")

#: production step-4 column order (cugen.gwas, linear)
SUMSTATS_COLUMNS = ("CHR", "POS", "ID", "REF", "ALT", "BETA", "SE", "Z", "P", "MAF", "gidx")


def _log(verbose, *a):
    if verbose:
        logger.info(" ".join(str(x) for x in a))


def _auto_block_step4(n, budget, n_file, dev):
    """Variants per block: ~10 % of the budget for the block working set, capped."""
    per = n * 8 * 3 + ((n_file + 3) // 4) * 4 * 2
    cap = 65536 if dev.type == "cuda" else 16384
    return int(max(512, min(cap, (budget * 0.10) // max(per, 1))))


def estimate_step4(n, M, K=1, block_size=8192, n_file=None, str_bytes_per_row=24):
    """Peak bytes of gwas_blocked (reader metadata + interpreter baseline are added on CPU).

    block phase : decode + G0/V float64 blocks + per-variant raw results (BETA/SE/Z float32 per
                  background, MAF)
    assembly    : raw results + shared annotation columns (CHR, POS, gidx, ID/REF/ALT strings,
                  str_bytes_per_row measured from the reader) + per-background result columns
                  + erfc temporaries
    """
    n_file = n_file or n
    block_size = max(1, min(block_size, M))
    blk = block_size * n * (8 * 2 + 1 + 1) + block_size * ((n_file + 3) // 4) * (4 + 1) * 2
    raw = M * (K * 12 + 4)
    ann = M * (8 + 8 + 8 + 8 + str_bytes_per_row)            # CHR POS gidx rows + strings
    cols = M * K * (4 + 4 + 4 + 8 + 4) + M * 16              # BETA SE Z P MAF per background + erfc temps
    peak = int(raw + max(blk, ann + cols) + 64e6)
    st = {"decode": float(M * n * 6), "gemm": 2.0 * 2 * M * n * K}
    return {"method": "gwas_blocked", "n": n, "M": M, "K": K, "block_size": block_size,
            "peak_bytes": peak, "peak_breakdown": {"block": blk, "raw_results": raw, "annotation": ann,
                                                   "result_columns": cols},
            "flops": float(sum(st.values())), "flops_by_stage": st, "dominant": "decode"}


def _as_configs(loco):
    """-> (list of (name, predictions (n, C), y_original (n,), fid_order, chromosomes (C,)), single)."""
    from .background.loco import LocoPredictions
    if isinstance(loco, (str, os.PathLike)):
        loco = LocoPredictions.load(loco)
    if isinstance(loco, LocoPredictions):
        if len(loco) == 1:
            return [(loco.phenotypes[0], loco.predictions[:, :, 0], loco.y_original[:, 0], loco.fid_order,
                     loco.chromosomes)], True
        return [(nm, lp.predictions[:, :, 0], lp.y_original[:, 0], lp.fid_order, lp.chromosomes)
                for nm, lp in loco.items()], False
    if isinstance(loco, dict):
        out = []
        for nm, v in loco.items():
            cf, _ = _as_configs(v)
            out += [(nm if len(cf) == 1 else f"{nm}/{c[0]}",) + c[1:] for c in cf]
        return out, False
    out = []
    for i, v in enumerate(loco):
        cf, _ = _as_configs(v)
        out += [(f"{i}:{c[0]}",) + c[1:] for c in cf]
    return out, False


def _annotation_columns(geno, rows):
    """CHR/POS/ID/REF/ALT (arrow-backed strings) and gidx for `rows`, built once and shared (not
    copied) by every result DataFrame."""
    import pandas as pd
    import pyarrow as pa
    rows = np.asarray(rows)
    front = {"CHR": geno.chrom[rows].astype(np.int64), "POS": geno.pos[rows].astype(np.int64)}
    ids = geno.variant_ids(pa.array(rows))
    cnt, oth = geno.alleles(pa.array(rows))
    arr = lambda a: pd.arrays.ArrowExtensionArray(pa.chunked_array(a) if isinstance(a, pa.Array) else a)  # noqa: E731
    if ids is not None:
        front["ID"] = arr(ids)
    if cnt is not None:
        front["REF"] = arr(oth)
        front["ALT"] = arr(cnt)
    back = {"gidx": geno.gidx[rows].astype(np.int64)}
    return front, back


def _frame(front, stats, back, counted_label):
    """One DataFrame from column dicts without copying the arrays (no concat)."""
    import pandas as pd
    cols = dict(front)
    cols.update(stats)
    cols.update(back)
    df = pd.DataFrame(cols, copy=False)
    df.attrs["counted_allele"] = counted_label
    return df


def _str_bytes_per_row(geno):
    """Average bytes per variant of the ID/REF/ALT string columns the output will carry."""
    tot = 0
    for c in (geno.variant_ids(),) + tuple(geno.alleles()):
        if c is not None:
            tot += c.nbytes
    return tot / max(geno.n_variants, 1)


def gwas_blocked(geno, loco, *, snps=None, chromosomes=None, maf_min=1e-4, stats="auto",
                 block_size=None, device="auto", memory_budget=None, estimate_only=False, output=None,
                 prefetch=2, check_samples=True, verbose=True):
    """Blocked linear step 4 on CPU or GPU: offset Wald test of every variant against one or
    several LOCO backgrounds.

    This is the engine ``cugen.gwas`` will dispatch to for ``engine="blocked"`` (CPU, PLINK
    .bed input, several backgrounds in one pass). It reproduces the production linear test
    (``cugen.gwas``, test="wald") to float32 precision.

    geno        : BedGenotypes / CugenGenotypes (same samples, same order as the backgrounds)
    loco        : LocoPredictions (one or several phenotypes), path to loco_predictions.npz, or
                  a dict / list of those (several backgrounds scored in one pass)
    snps        : variants to test (IDs / path / mask / rows); None = all
    chromosomes : chromosomes to test (default: all; every tested chromosome must be a column
                  of each background)
    maf_min     : drop variants with MAF below this (default 1e-4, as ``cugen.gwas``; None keeps all)
    stats       : "stored" (cugen stored mu_x/sxx/maf, the package behaviour) | "recompute" |
                  "auto" (stored for cugens, recompute otherwise)
    output      : optional path. A directory (or a path without extension) gets the production
                  layout ``chr{c}_sumstats.tsv`` + ``all_sumstats.tsv.gz`` (one subdirectory per
                  background when there are several); a file ending in ``.tsv``/``.tsv.gz``
                  (exact integer columns, ``%.6g`` floats), ``.feather`` or ``.parquet`` gets one
                  table (single background only).
    check_samples : require the backgrounds' ``fid_order`` to equal ``geno.sample_ids``.

    Returns a DataFrame (one background) or a dict name -> DataFrame with columns
    ``CHR POS ID REF ALT BETA SE Z P MAF gidx``; ``df.attrs["counted_allele"]`` names the
    allele BETA counts (ALT).
    """
    dev = _device(device)
    with dev:
        return _gwas_blocked(dev, geno, loco, snps=snps, chromosomes=chromosomes, maf_min=maf_min,
                             stats=stats, block_size=block_size, memory_budget=memory_budget,
                             estimate_only=estimate_only, output=output, prefetch=prefetch,
                             check_samples=check_samples, verbose=verbose)


def _gwas_blocked(dev, geno, loco, *, snps, chromosomes, maf_min, stats, block_size, memory_budget,
                  estimate_only, output, prefetch, check_samples, verbose):
    xp = dev.xp
    if stats not in ("auto", "stored", "recompute"):
        raise ValueError("stats must be 'auto', 'stored' or 'recompute'")
    cfgs, single = _as_configs(loco)
    K = len(cfgs)
    n = geno.n
    gi = np.asarray(geno.sample_ids).astype(str)
    named = not np.array_equal(gi, np.arange(n).astype(str))
    for nm, p, yo, fid, _ in cfgs:
        if p.shape[0] != n:
            raise ValueError(f"LOCO background {nm}: {p.shape[0]} samples, genotypes have {n}")
        if check_samples and named and fid is not None and not np.array_equal(np.asarray(fid).astype(str), gi):
            raise ValueError(f"LOCO background {nm}: fid_order differs from the genotype sample order "
                             f"(first {list(np.asarray(fid)[:3])} vs {list(gi[:3])}); align the samples, or "
                             f"pass check_samples=False if only the ID naming differs (e.g. FID vs IID)")
    rows = geno.select(snps, chromosomes=chromosomes)
    M = len(rows)
    tested = sorted({int(c) for c in np.unique(geno.chrom[rows])}) if M else []
    for nm, _, _, _, ch in cfgs:
        lack = sorted(set(tested) - {int(c) for c in ch})
        if lack:
            raise ValueError(f"genotypes include chromosome(s) {lack} but LOCO background {nm} has columns "
                             f"for chromosomes {ch.tolist()}; pass chromosomes={ch.tolist()} or refit the "
                             f"background on those chromosomes")
    if stats == "auto":
        stats = "stored" if geno.stored_stats(rows[:1]) is not None else "recompute"
    if stats == "stored" and geno.stored_stats(rows[:1]) is None:
        raise ValueError("stats='stored' needs cugen input (stored mu_x/sxx/maf)")
    bud = R.budget(dev, memory_budget)
    bsz = block_size or _auto_block_step4(n, bud, geno.n_file, dev)
    est = R.finish(estimate_step4(n, M, K, bsz, geno.n_file, _str_bytes_per_row(geno)), dev, memory_budget,
                   [f"block_size={max(bsz // 4, 256)}", "fewer LOCO backgrounds per call",
                    "chromosomes=[...] (run chromosomes in separate calls)", "write to output= per chromosome"],
                   "gwas_blocked", geno)
    if estimate_only:
        return est
    R.check(est, "gwas_blocked")
    _log(verbose, dev.describe(), f"| n={n} M={M:,} backgrounds={K} block={bsz} stats={stats} | "
         f"est. peak {est['peak']}")
    _log(verbose, f"BETA per copy of the counted allele: {geno.counted_label}")
    t0 = time.time()
    beta = np.empty((M, K), np.float32)
    se = np.empty((M, K), np.float32)
    z = np.empty((M, K), np.float32)
    mafv = np.empty(M, np.float32)
    lut_f = xp.asarray(np.array([0.0, 1.0, 2.0, 0.0]))
    lut_v = xp.asarray(np.array([1.0, 1.0, 1.0, 0.0]))
    t_dec = t_gemm = 0.0
    for c, s0, e0 in geno.chrom_ranges(rows):
        cols = [int(np.flatnonzero(ch == c)[0]) for _, _, _, _, ch in cfgs]
        Rm = np.stack([yo - p[:, j] for (_, p, yo, _, _), j in zip(cfgs, cols)], 1).astype(np.float64)
        Rm = Rm - Rm.mean(0)
        syy = (Rm * Rm).sum(0)
        Rg = dev.asarray(Rm)
        syy_g = dev.asarray(syy)
        blocks = [(c, s, min(s + bsz, e0)) for s in range(s0, e0, bsz)]
        for _, s, e, counts in geno.iter_blocks(rows, blocks, dev, prefetch):
            td = time.time()
            G0 = lut_f[counts]                                   # dosage, missing -> 0
            miss = counts == 3
            anym = bool(miss.any())
            V = lut_v[counts] if anym else None
            if stats == "stored":
                mu, sxx, maf = geno.stored_stats(rows[s:e])
                mu = dev.asarray(mu.astype(np.float64))
                sxx = dev.asarray(sxx.astype(np.float64))
            else:
                nval = (~miss).sum(1).astype(xp.float64)
                sg = G0.sum(1)
                sg2 = (G0 * G0).sum(1)
                mu = sg / xp.maximum(nval, 1)
                sxx = sg2 - nval * mu * mu
                maf = dev.asnumpy(xp.minimum(mu / 2, 1 - mu / 2)).astype(np.float32)
            del counts, miss
            t_dec += time.time() - td
            tg = time.time()
            num = G0 @ Rg                                         # sum_valid g r
            if anym:
                num -= mu[:, None] * (V @ Rg)                     # - mu sum_valid r
            else:
                num -= mu[:, None] * Rg.sum(0)[None, :]
            den = xp.maximum(sxx, 1e-20)[:, None]
            b = num / den
            sev = xp.sqrt(xp.maximum(syy_g[None, :] - b * b * den, 1e-20) / ((n - 2) * den))
            zz = b / xp.maximum(sev, 1e-20)
            beta[s:e] = dev.asnumpy(b)
            se[s:e] = dev.asnumpy(sev)
            z[s:e] = dev.asnumpy(zz)
            mafv[s:e] = maf
            t_gemm += time.time() - tg
            del G0, V, num, b, sev, zz
    dev.free_pool()
    T = {"decode_stats_s": t_dec, "gemm_test_s": t_gemm, "total_s": time.time() - t0}
    _log(verbose, f"scored {M:,} variants x {K} backgrounds in {T['total_s']:.1f}s "
         f"(decode/stats {t_dec:.1f}s, gemm/test {t_gemm:.1f}s)")
    import scipy.special as ssp
    keep = None
    if maf_min is not None and maf_min > 0:
        keep = mafv >= maf_min
    res = {}
    sl = slice(None) if keep is None else keep
    front, back = _annotation_columns(geno, rows if keep is None else rows[keep])
    maf_col = mafv[sl] if keep is not None else mafv
    for k, (nm, _, _, _, _) in enumerate(cfgs):
        zk = z[sl, k]
        st_cols = {"BETA": beta[sl, k], "SE": se[sl, k], "Z": zk,
                   "P": ssp.erfc(np.abs(zk, dtype=np.float64) / np.sqrt(2.0)), "MAF": maf_col}
        df = _frame(front, st_cols, back, geno.counted_label)
        df.attrs["timing_s"] = T
        df.attrs["stats"] = stats
        res[nm] = df
    if output is not None:
        write_sumstats(res, output, single)
    return res[cfgs[0][0]] if single else res


def _write_table(df, path):
    import pyarrow as pa
    from .assoc import _write_sumstats
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    attrs, df.attrs = df.attrs, {}
    try:
        if path.endswith(".parquet"):
            df.to_parquet(path, index=False)
        elif path.endswith(".tsv"):
            _write_sumstats(df.copy(), path)
        elif path.endswith(".tsv.gz"):
            tmp = path[:-3]
            _write_sumstats(df.copy(), tmp)
            with open(tmp, "rb") as src, gzip.open(path, "wb", compresslevel=6) as dst:
                shutil.copyfileobj(src, dst, 1 << 22)
            os.remove(tmp)
        else:
            import pyarrow.feather as pf
            pf.write_feather(pa.Table.from_pandas(df, preserve_index=False), path)
    finally:
        df.attrs = attrs


def _write_layout(df, out_dir):
    """Production step-4 layout: chr{c}_sumstats.tsv per chromosome + all_sumstats.tsv.gz."""
    from .assoc import _write_sumstats
    os.makedirs(out_dir, exist_ok=True)
    files = []
    for c in sorted(df["CHR"].unique()):
        f = os.path.join(out_dir, f"chr{int(c)}_sumstats.tsv")
        _write_sumstats(df[df["CHR"] == c].reset_index(drop=True), f)
        files.append(f)
    with gzip.open(os.path.join(out_dir, "all_sumstats.tsv.gz"), "wb", compresslevel=6) as dst:
        for i, f in enumerate(files):
            with open(f, "rb") as src:
                if i:
                    src.readline()                     # header once
                shutil.copyfileobj(src, dst, 1 << 22)


def write_sumstats(res, output, single=None):
    """Write gwas_blocked results (DataFrame or dict) to ``output`` (see :func:`gwas_blocked`)."""
    if not isinstance(res, dict):
        res, single = {"y": res}, True
    single = len(res) == 1 if single is None else single
    output = str(output)
    is_file = not os.path.isdir(output) and output.endswith((".tsv", ".tsv.gz", ".feather", ".parquet"))
    if is_file:
        if len(res) != 1:
            raise ValueError("several backgrounds need a directory output")
        _write_table(next(iter(res.values())), output)
        return
    for nm, df in res.items():
        d = output if single else os.path.join(output, nm.replace("/", "_").replace(":", "_"))
        _write_layout(df, d)
