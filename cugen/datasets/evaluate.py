"""cugen.datasets.evaluate - score GWAS output against a simulated truth (CPU).

    evaluate_gwas(sumstats, data_dir)     lambda, GWS loci, recovered / false loci
    compare_sumstats(a, b)                r(z), slope, loci overlap between two runs
    background_r2(run_dir, data_dir)      in-sample vs held-out R2 of the step-3 model
    load_sumstats(path) / load_regenie(path)

Loci are distance clumps (``cugen.loci.distance_clumps``): genome-wide
significant variants in order of significance, a new lead whenever no earlier
lead on the same chromosome is within ``clump_kb``.
A causal variant is *recovered* when a lead lies within ``recover_kb`` of it; a
locus is a *false positive* when no causal variant lies within ``fp_kb`` of its
lead. ``lambda_null`` is the genomic control over variants in LD blocks that
contain no causal variant (blocks are independent in the simulator), so it
measures calibration without the polygenic signal inflating it.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..loci import distance_clumps, lambda_gc


def _has_z(df):
    return "Z" in df.columns and df["Z"].notna().any()


def _chi2_from(df):
    if _has_z(df):
        z = df["Z"].to_numpy(dtype=np.float64)
        return z * z
    from scipy.stats import chi2
    return chi2.isf(df["P"].to_numpy(dtype=np.float64), 1)


def _lambda(df, mask=None):
    """lambda_GC from Z when present, else from P (one constant, ``cugen.loci.lambda_gc``)."""
    if _has_z(df):
        z = df["Z"].to_numpy(dtype=np.float64)
        return lambda_gc(z=z if mask is None else z[mask])
    p = df["P"].to_numpy(dtype=np.float64)
    return lambda_gc(p=p if mask is None else p[mask])


def load_sumstats(path):
    """CuGen sumstats (tsv / tsv.gz / feather) -> DataFrame."""
    path = str(path)
    if path.endswith(".feather"):
        return pd.read_feather(path)
    return pd.read_csv(path, sep="\t")


def load_regenie(path, annotation=None):
    """REGENIE step-2 ``.regenie`` -> DataFrame with CuGen column names.

    Z = BETA / SE, P = 10^-LOG10P. If ``annotation`` (feather with ID / ALT /
    gidx) is given, BETA and Z are flipped where REGENIE's tested allele
    (ALLELE1) is not the annotation ALT, and gidx is attached.
    """
    df = pd.read_csv(path, sep=r"\s+")
    out = pd.DataFrame({"CHR": df["CHROM"].astype(int), "POS": df["GENPOS"].astype(np.int64),
                        "ID": df["ID"].astype(str), "ALLELE1": df["ALLELE1"].astype(str),
                        "BETA": df["BETA"].astype(float), "SE": df["SE"].astype(float),
                        "P": np.power(10.0, -df["LOG10P"].astype(float)),
                        "A1FREQ": df["A1FREQ"].astype(float)})
    out["Z"] = out["BETA"] / out["SE"]
    if annotation is not None:
        ann = pd.read_feather(annotation, columns=["gidx", "ID", "ALT"])
        out = out.merge(ann, on="ID", how="left")
        flip = out["ALLELE1"].to_numpy() != out["ALT"].to_numpy()
        for c in ("BETA", "Z"):
            out.loc[flip, c] = -out.loc[flip, c]
        out["A1FREQ"] = np.where(flip, 1 - out["A1FREQ"], out["A1FREQ"])
    return out


def clump_loci(df, p_threshold=5e-8, clump_kb=500):
    """Distance clumping of GWS variants -> DataFrame of leads (CHR, POS, P, ID, gidx, n_gws).

    Variants are ranked by P (ties keep position order) and clumped with
    ``cugen.loci.distance_clumps``: a variant within ``clump_kb`` of an earlier lead joins
    that lead, otherwise it starts a new locus.
    """
    cols = ["CHR", "POS", "P", "ID", "gidx", "n_gws"]
    g = df[df["P"] < p_threshold]
    if not len(g):
        return pd.DataFrame(columns=cols)
    p = g["P"].to_numpy(dtype=np.float64)
    with np.errstate(divide="ignore"):
        score = -np.log10(p)                      # P == 0 -> inf (ties keep position order)
    chrom = g["CHR"].to_numpy()
    pos = g["POS"].to_numpy()
    cl = distance_clumps(chrom, pos, score, window=int(clump_kb * 1000))
    if "row" in cl.columns:
        rows = cl["row"].to_numpy(dtype=np.int64)
    else:                                         # match leads back by (CHR, POS, score)
        key = pd.DataFrame({"CHR": chrom, "POS": pos, "s": score, "row": np.arange(len(g))})
        rows = cl.merge(key.drop_duplicates(["CHR", "POS"]), on=["CHR", "POS"], how="left")["row"].to_numpy()
    lead = g.iloc[rows]
    out = pd.DataFrame({
        "CHR": lead["CHR"].astype(int).to_numpy(), "POS": lead["POS"].astype(np.int64).to_numpy(),
        "P": lead["P"].astype(float).to_numpy(),
        "ID": lead["ID"].astype(str).to_numpy() if "ID" in lead.columns else np.full(len(lead), ""),
        "gidx": lead["gidx"].astype(np.int64).to_numpy() if "gidx" in lead.columns else np.full(len(lead), -1),
        "n_gws": cl["n_sig"].astype(int).to_numpy()})
    return out.sort_values(["CHR", "P", "POS"], kind="stable").reset_index(drop=True)[cols]


def _nearest(chrom_a, pos_a, chrom_b, pos_b):
    """For each a, distance (bp) to the nearest b on the same chromosome (inf if none)."""
    d = np.full(len(pos_a), np.inf)
    for c in np.unique(chrom_a):
        ia = np.flatnonzero(chrom_a == c)
        pb = np.sort(pos_b[chrom_b == c])
        if pb.size == 0:
            continue
        k = np.searchsorted(pb, pos_a[ia])
        lo = np.abs(pos_a[ia] - pb[np.clip(k - 1, 0, pb.size - 1)])
        hi = np.abs(pos_a[ia] - pb[np.clip(k, 0, pb.size - 1)])
        d[ia] = np.minimum(lo, hi)
    return d


def _truth(data_dir):
    d = Path(data_dir)
    truth = pd.read_csv(d / "truth.tsv", sep="\t")
    blocks = pd.read_csv(d / "ld_blocks.tsv", sep="\t") if (d / "ld_blocks.tsv").exists() else None
    return truth, blocks


def evaluate_gwas(sumstats, data_dir, *, p_threshold=5e-8, recover_kb=250, fp_kb=500,
                  clump_kb=500, return_loci=False):
    """Score a sumstats table (path or DataFrame) against ``data_dir``/truth.tsv."""
    df = load_sumstats(sumstats) if isinstance(sumstats, (str, Path)) else sumstats
    df = df[np.isfinite(df["P"].to_numpy(dtype=np.float64))]
    truth, blocks = _truth(data_dir)
    chi2 = _chi2_from(df)
    res = dict(n_tested=int(len(df)), lambda_gc=_lambda(df), mean_chi2=float(np.nanmean(chi2)))
    if blocks is not None and "gidx" in df.columns:
        gi = df["gidx"].to_numpy(dtype=np.int64)
        bi = np.searchsorted(blocks["gidx_start"].to_numpy(), gi, side="right") - 1
        null = blocks["n_causal"].to_numpy()[bi] == 0
        res["lambda_null"] = _lambda(df, null)
        res["n_null"] = int(null.sum())
        res["n_gws_null_variants"] = int((df["P"].to_numpy()[null] < p_threshold).sum())
    loci = clump_loci(df, p_threshold, clump_kb)
    res["n_gws_variants"] = int((df["P"] < p_threshold).sum())
    res["n_loci"] = int(len(loci))
    tc, tp = truth["CHR"].to_numpy(), truth["POS"].to_numpy()
    if len(loci):
        dl = _nearest(loci["CHR"].to_numpy(), loci["POS"].to_numpy(), tc, tp)
        loci["dist_causal_kb"] = dl / 1000.0
        loci["false_positive"] = dl > fp_kb * 1000
        # nearest causal is large?
        lg = truth[truth["is_large"] == 1]
        dlg = _nearest(loci["CHR"].to_numpy(), loci["POS"].to_numpy(),
                       lg["CHR"].to_numpy(), lg["POS"].to_numpy())
        loci["large_locus"] = dlg <= recover_kb * 1000
        dc = _nearest(tc, tp, loci["CHR"].to_numpy(), loci["POS"].to_numpy())
    else:
        dc = np.full(len(truth), np.inf)
    # strict false positives: lead in an LD block without any causal variant (blocks are
    # simulated independently, so such a lead has no LD with any causal variant)
    if len(loci) and blocks is not None and (loci["gidx"] >= 0).all():
        lb = np.searchsorted(blocks["gidx_start"].to_numpy(), loci["gidx"].to_numpy(), side="right") - 1
        loci["null_block"] = blocks["n_causal"].to_numpy()[lb] == 0
        res["n_loci_null_block"] = int(loci["null_block"].sum())
    rec = dc <= recover_kb * 1000
    lg = truth["is_large"].to_numpy() == 1
    res.update(n_large=int(lg.sum()), n_large_recovered=int((rec & lg).sum()),
               n_poly=int((~lg).sum()), n_poly_recovered=int((rec & ~lg).sum()),
               var_explained_recovered=float(truth.loc[rec, "var_explained"].sum()),
               n_false_loci=int(loci["false_positive"].sum()) if len(loci) else 0)
    # causal variants that are themselves GWS
    if "gidx" in df.columns:
        gws = set(df.loc[df["P"] < p_threshold, "gidx"].astype(np.int64))
        res["n_causal_variants_gws"] = int(sum(int(g) in gws for g in truth["gidx"]))
    res["params"] = dict(p_threshold=p_threshold, recover_kb=recover_kb, fp_kb=fp_kb, clump_kb=clump_kb)
    return (res, loci) if return_loci else res


def compare_sumstats(a, b, *, on="ID", p_threshold=5e-8, clump_kb=500, match_kb=250):
    """r(z) and slope (b on a) over shared variants, plus GWS-locus overlap."""
    A = load_sumstats(a) if isinstance(a, (str, Path)) else a
    B = load_sumstats(b) if isinstance(b, (str, Path)) else b
    m = A[[on, "Z", "P", "CHR", "POS"]].merge(B[[on, "Z", "P"]], on=on, suffixes=("_a", "_b"))
    m = m[np.isfinite(m["Z_a"]) & np.isfinite(m["Z_b"])]
    za, zb = m["Z_a"].to_numpy(), m["Z_b"].to_numpy()
    r = float(np.corrcoef(za, zb)[0, 1])
    slope = float(np.dot(za - za.mean(), zb - zb.mean()) / np.dot(za - za.mean(), za - za.mean()))
    la = clump_loci(A, p_threshold, clump_kb)
    lb = clump_loci(B, p_threshold, clump_kb)
    if len(la) and len(lb):
        da = _nearest(la["CHR"].to_numpy(), la["POS"].to_numpy(), lb["CHR"].to_numpy(), lb["POS"].to_numpy())
        db = _nearest(lb["CHR"].to_numpy(), lb["POS"].to_numpy(), la["CHR"].to_numpy(), la["POS"].to_numpy())
        a_in_b, b_in_a = int((da <= match_kb * 1000).sum()), int((db <= match_kb * 1000).sum())
    else:
        a_in_b = b_in_a = 0
    return dict(n_shared=int(len(m)), r_z=r, slope_b_on_a=slope, n_loci_a=int(len(la)),
                n_loci_b=int(len(lb)), loci_a_in_b=a_in_b, loci_b_in_a=b_in_a)


# --------------------------------------------------------------------------
# background (step-3 model) R2
# --------------------------------------------------------------------------
def _cugen_meta(path):
    """(n_samples, bytes_per_variant, data_offset, gidx) from the file header (CPU only)."""
    from ..io import CugenReader
    r = CugenReader(path)
    try:
        return int(r.n_samples), int(r.bytes_per_variant), int(r.data_offset), np.asarray(r.gidx)
    finally:
        r.close()


def _read_cugen_rows(path, local_idx, meta=None):
    """(n_samples, k) float32 dosages of variant rows ``local_idx`` (NaN = missing)."""
    from .._codec import MISSING, unpack_cugen
    n, bpv, off, _ = meta or _cugen_meta(path)
    idx = np.asarray(local_idx, dtype=np.int64)
    raw = np.empty((len(idx), bpv), dtype=np.uint8)
    with open(path, "rb") as f:
        fd = f.fileno()
        for k, i in enumerate(idx):
            b = os.pread(fd, bpv, off + int(i) * bpv)
            if len(b) != bpv:
                raise IOError(f"{path}: short read at variant {int(i)}")
            raw[k] = np.frombuffer(b, dtype=np.uint8)
    codes = unpack_cugen(raw, n)
    x = codes.T.astype(np.float32)
    x[codes.T == MISSING] = np.nan
    return x


def _active_dosages(cugen_dir, model):
    """(n, k) float32 dosages of the model's SNPs, in model row order."""
    cols = []
    for chrom, sub in model.groupby("chr_num", sort=False):
        path = os.path.join(cugen_dir, f"chr{int(chrom)}.cugen")
        meta = _cugen_meta(path)
        loc = np.searchsorted(meta[3], sub["gidx"].to_numpy())
        cols.append((sub.index.to_numpy(), _read_cugen_rows(path, loc, meta)))
    order = np.concatenate([c[0] for c in cols])
    X = np.concatenate([c[1] for c in cols], axis=1)
    return X[:, np.argsort(order)] if len(order) else X


def _prs(cugen_dir, model, n):
    s = np.zeros(n)
    for chrom, sub in model.groupby("chr_num"):
        path = os.path.join(cugen_dir, f"chr{int(chrom)}.cugen")
        meta = _cugen_meta(path)
        loc = np.searchsorted(meta[3], sub["gidx"].to_numpy())
        mu = sub["snp_mean"].to_numpy(np.float32)
        b = sub["step3_coef"].to_numpy(np.float64)
        for k in range(0, len(loc), 2000):
            x = _read_cugen_rows(path, loc[k:k + 2000], meta)
            x = np.nan_to_num(x - mu[k:k + 2000][None, :])
            s += x.astype(np.float64) @ b[k:k + 2000]
    return s


def _resid(y, C, coef=None):
    X = np.column_stack([np.ones(len(y)), C])
    if coef is None:
        coef = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ coef, coef


def background_r2(run_dir, data_dir, *, covariates=None, ridge=1e-2, refit=True):
    """In-sample vs held-out R2 of the step-3 joint-LASSO model (SNP part).

    PRS = sum over active SNPs of (x - snp_mean) * step3_coef, scored on the
    training cugens and on ``data_dir/test/cugen``; R2 = corr(PRS, height
    residualised on the covariates)^2 (test residualised with the training
    covariate fit). Also returns the in-sample R2 of each LOCO prediction and
    the R2 of the TRUE genetic value on the test set (the ceiling).

    ``refit`` (default True) adds ``r2_refit_insample`` / ``r2_refit_heldout``: the
    step-3 LOCO refit's model (ridge OLS, penalty ``ridge`` * n on the training-
    standardised active SNPs) fitted on all active SNPs against the covariate-
    residualised training height, scored in-sample and on the test cohort. This
    measures the overfit of the unshrunk refit that produces the LOCO offsets (the
    step3_coef PRS above is the shrunk LASSO fit).
    """
    run, d = Path(run_dir), Path(data_dir)
    model = pd.read_feather(run / "full_model.feather")
    model = model[model["step3_active"]].sort_values("gidx")
    ph = pd.read_csv(d / "pheno.tsv", sep="\t", dtype={"FID": str})
    cov = covariates or [c for c in ph.columns if c in ("age", "sex") or c.startswith("PC")]
    yr, coef = _resid(ph["height"].to_numpy(float), ph[cov].to_numpy(float))
    prs = _prs(str(d / "cugen"), model, len(ph))
    out = dict(n_active=int(len(model)), r2_insample=float(np.corrcoef(prs, yr)[0, 1] ** 2))
    if (d / "test" / "cugen").exists():
        pt = pd.read_csv(d / "test" / "pheno.tsv", sep="\t", dtype={"FID": str})
        yt, _ = _resid(pt["height"].to_numpy(float), pt[cov].to_numpy(float), coef)
        prs_t = _prs(str(d / "test" / "cugen"), model, len(pt))
        out["r2_heldout"] = float(np.corrcoef(prs_t, yt)[0, 1] ** 2)
        if refit and len(model):
            m2 = model.reset_index(drop=True)
            Xtr = _active_dosages(str(d / "cugen"), m2).astype(np.float64)
            mu, sd = Xtr.mean(0), Xtr.std(0)
            sd[sd < 1e-6] = 1.0
            Xtr = (Xtr - mu) / sd
            n = Xtr.shape[0]
            b = np.linalg.solve(Xtr.T @ Xtr + ridge * n * np.eye(Xtr.shape[1]), Xtr.T @ (yr - yr.mean()))
            out["r2_refit_insample"] = float(np.corrcoef(Xtr @ b, yr)[0, 1] ** 2)
            del Xtr
            Xte = (_active_dosages(str(d / "test" / "cugen"), m2).astype(np.float64) - mu) / sd
            out["r2_refit_heldout"] = float(np.corrcoef(Xte @ b, yt)[0, 1] ** 2)
        gv = np.load(d / "genetic_value.npz", allow_pickle=True)
        gt = gv["g"][gv["split"] == 1]
        out["r2_true_g_heldout"] = float(np.corrcoef(gt, yt)[0, 1] ** 2)
    lp = run / "loco_predictions.npz"
    if lp.exists():
        z = np.load(lp, allow_pickle=True)
        P, y = z["predictions"].astype(np.float64), z["y_original"].astype(np.float64)
        sst = ((y - y.mean()) ** 2).sum()
        r2 = 1 - ((y[:, None] - P) ** 2).sum(0) / sst
        out.update(loco_r2_insample_mean=float(r2.mean()), loco_r2_insample_min=float(r2.min()),
                   loco_r2_insample_max=float(r2.max()))
    return out


def evaluate_finemapping(fm_dir, data_dir, pip_threshold=0.5):
    """Score UltraSuSiE output (``<run>/finemapping``) against the simulated truth.

    A credible set is (locus, effect_idx) from ``locus_*/susie_credible_sets.tsv``.
    Returns CS count / sizes, the fraction of CSs that contain a causal variant,
    distinct causal variants captured, PIP > ``pip_threshold`` precision, and how many
    large-effect causal variants sit in a CS.
    """
    fm = Path(fm_dir)
    truth, _ = _truth(data_dir)
    causal = set(truth["gidx"].astype(np.int64))
    large = set(truth.loc[truth["is_large"] == 1, "gidx"].astype(np.int64))
    parts = []
    for p in sorted(fm.glob("locus_*/susie_credible_sets.tsv")):
        t = pd.read_csv(p, sep="\t")
        if len(t):
            parts.append(t.assign(locus=p.parent.name))
    out = dict(n_loci_finemapped=int(len(list(fm.glob("locus_*")))))
    if parts:
        cs = pd.concat(parts, ignore_index=True)
        cs["is_causal"] = cs["gidx"].astype(np.int64).isin(causal)
        g = cs.groupby(["locus", "effect_idx"])
        size = g.size()
        hit = g["is_causal"].any()
        out.update(n_cs=int(len(size)), cs_size_median=float(size.median()),
                   cs_size_mean=float(size.mean()), n_cs_size1=int((size == 1).sum()),
                   frac_cs_with_causal=float(hit.mean()),
                   n_causal_in_cs=int(len(set(cs.loc[cs["is_causal"], "gidx"].astype(np.int64)))),
                   n_large_in_cs=int(len(large & set(cs["gidx"].astype(np.int64)))))
    av = fm / "all_variants.tsv.gz"
    if av.exists():
        v = pd.read_csv(av, sep="\t", usecols=["gidx", "susie_pip"])
        hi = v[v["susie_pip"] > pip_threshold]
        out.update(n_pip_above=int(len(hi)),
                   frac_pip_above_causal=float(hi["gidx"].astype(np.int64).isin(causal).mean()) if len(hi) else None,
                   sum_pip=float(v["susie_pip"].sum()),
                   n_causal_tested=int(v["gidx"].astype(np.int64).isin(causal).sum()))
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="score sumstats against a simulated truth")
    ap.add_argument("sumstats")
    ap.add_argument("data_dir")
    a = ap.parse_args(argv)
    print(json.dumps(evaluate_gwas(a.sumstats, a.data_dir), indent=2))


if __name__ == "__main__":
    main()
