"""cugen.prs — polygenic-score (PRS) workflow + composable PRS building blocks.

This module packages the manuscript train/val/test PRS recipe (sessions 41-42). It is exposed
two ways in the :mod:`cugen.ultralasso` namespace, so you can either run the whole pipeline OR
script your own with the individual pieces:

**1. One-call pipeline** ::

    from cugen import ultralasso
    ultralasso.build_prs_splits(ref_npz=..., master_phe=..., source_cugen_dir=...,
                                phenotypes=[("INI50", False)], splits={...},
                                out_dir=..., scratch_dir=...)
    res = ultralasso.prs("INI50", splits_dir=..., cohorts_dir=..., score_cugen_dir=...,
                         train_cugen_dir=..., trainval_cugen_dir=..., ref_npz=...,
                         master_phe=..., step2_alpha=2e-4, out_dir=...)

**2. Composable building blocks** (program your own scripts) ::

    cands  = ultralasso.prs_screen(train_cugen_dir, train_npz, step2_alpha=2e-4)
    sweep  = ultralasso.prs_alpha_sweep(cands, train_npz, train_cugen_dir, score_cugen_dir,
                                        alphas=[6e-3, 1e-2, 1.5e-2], y=y, Xcov=Xcov,
                                        idx_val=idx_val, idx_train=idx_train)
    best   = ultralasso.select_best_alpha(sweep)            # argmax validation metric
    w      = ultralasso.prs_fit_weights(cands, trainval_npz, trainval_cugen_dir, best)
    prs_v  = ultralasso.prs_score(w, score_cugen_dir)       # PRS for every sample
    r2c, r2f, inc, n = ultralasso.prs_r2(y[idx_test], Xcov[idx_test], prs_v[idx_test])

UltraLasso *is* uniLasso: step1+2 builds per-variant leave-one-out univariate predictions and
runs a non-negative LASSO on them (``cg.screen_chromosome``); step3 is the joint LASSO refit
(``cg.fit_joint_lasso``). The PRS is the joint active set + its coefficients, scored on the full
cohort by ``cg.score``. Compare to the preprint **uniLasso** row (NOT uniLasso-ES).

Everything here is pure orchestration over already-validated package functions
(``screen_chromosome``, ``fit_joint_lasso``, ``score``, ``subset_cugen_dir``) — no new
low-level compute. Heavy CuPy/sklearn imports are deferred into the functions so the module
imports cheaply on a login node.

Metrics:
  * continuous trait -> full-model R-squared (covariates + PRS), incremental over covariates
    (:func:`prs_r2`).
  * binary trait     -> full-model logistic AUC (covariates + PRS), incremental over covariates
    (:func:`prs_auc`); the predictor is fit on the LPM (0/1) joint refit (``y_original``).
"""
from __future__ import annotations

import struct
import time
from pathlib import Path
from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

__all__ = [
    "prs", "build_prs_splits",
    # composable building blocks
    "prs_load_phenotype", "prs_residualize", "prs_screen", "prs_fit_weights",
    "prs_score", "prs_alpha_sweep", "select_best_alpha", "prs_r2", "prs_auc",
    # default grids
    "DEFAULT_COVARS", "DEFAULT_STEP3_ALPHAS_CONT", "DEFAULT_STEP3_ALPHAS_BIN",
]

# ---------------------------------------------------------------------------------------------
# What the PRS scripts had to code OUTSIDE the cg library (glue beyond calling cg functions),
# and the building block that now owns each one — so you can script a custom PRS without
# re-deriving any of it:
#
#   script glue (sessions 41-42)                       -> cugen.ultralasso building block
#   ------------------------------------------------------------------------------------
#   load y + covariates in cohort order from master.phe -> prs_load_phenotype
#   per-split OLS residual (LPM for binary) + NPZ schema -> prs_residualize / build_prs_splits
#   read .cugen header for the half-split midpoint       -> prs_screen (internal)
#   genome-wide screen loop + chr1-11 half-split + concat-> prs_screen
#   cap candidates to top-N by strength (GPU-mem)        -> prs_screen(max_snps=...)
#   feather-cache the candidates                         -> prs_screen(cache_path=...)
#   extract active weights from fit_joint_lasso result   -> prs_fit_weights
#   score + sort by sample_idx -> per-sample PRS array    -> prs_score
#   incremental full-model R-squared                      -> prs_r2
#   incremental full-model logistic AUC                   -> prs_auc
#   alpha sweep loop + per-alpha resume checkpoint        -> prs_alpha_sweep
#   pick best alpha by validation metric                 -> select_best_alpha
# ---------------------------------------------------------------------------------------------

# Default Fix-I covariates (must match the cohort residualization).
DEFAULT_COVARS = ["age", "sex"] + [f"Global_PC{i}" for i in range(1, 11)]

# step3 (joint LASSO) alpha grids. The binary LPM outcome variance (~0.1) is ~30x smaller than a
# continuous trait (height ~84), so the continuous grid over-penalizes binary models down to
# ~empty. Use a ~30x-smaller default grid for binary.
DEFAULT_STEP3_ALPHAS_CONT = [6e-3, 8e-3, 1e-2, 1.2e-2, 1.5e-2, 2e-2]
DEFAULT_STEP3_ALPHAS_BIN = [6e-5, 1e-4, 2e-4, 4e-4, 8e-4, 1.5e-3]

# step2 (screen) FISTA alpha default. Continuous traits use 6e-4; binary LPM-residual traits need
# a MUCH smaller alpha and should pass a calibrated value. No safe universal binary default.
DEFAULT_STEP2_ALPHA_CONT = 6e-4

# step3 fit_joint_lasso builds an N_features^2 Gram matrix; at trainval (~269k samples) on an
# 80 GB GPU, >~50k candidates OOM. Cap to the top-N by screen 'strength'.
DEFAULT_MAX_STEP3_SNPS = 44000


# --------------------------------------------------------------------------- private helpers
def _cugen_nvariants(path: Union[str, Path]) -> int:
    with open(path, "rb") as f:
        h = f.read(64)
    return struct.unpack_from("<q", h, 24)[0]


def prs_residualize(y, X, binary=False):
    """OLS-residualize a phenotype on ``[1, X]`` (covariates). Binary: LPM recode 1->0, 2->1, -9->nan.

    The same residualization :func:`build_prs_splits` applies per split; exposed so you can build
    custom cohort NPZs. Returns ``(resid, yhat, y_used, mask_valid, coef, r2_cov)`` aligned to the
    input rows (``resid``/``yhat`` are NaN where the phenotype or a covariate is missing).
    """
    is_binary = binary
    y = np.asarray(y, dtype=np.float64).copy()
    if is_binary:
        y_re = np.full_like(y, np.nan)
        y_re[y == 1] = 0.0
        y_re[y == 2] = 1.0
        y_re[y == 0] = 0.0  # accept already-0/1 coding
        y = y_re
    else:
        y[y == -9] = np.nan
    mask = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
    Xd = np.column_stack([np.ones(mask.sum()), X[mask]])
    coef, *_ = np.linalg.lstsq(Xd, y[mask], rcond=None)
    yhat_valid = Xd @ coef
    yhat = np.full_like(y, np.nan)
    yhat[mask] = yhat_valid
    resid = np.full_like(y, np.nan)
    resid[mask] = y[mask] - yhat_valid
    ss_tot = np.sum((y[mask] - y[mask].mean()) ** 2)
    r2 = 1.0 - np.sum((y[mask] - yhat_valid) ** 2) / ss_tot if ss_tot > 0 else np.nan
    return resid, yhat, y, mask, coef, float(r2)


def _read_keep_fids(path):
    fids = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if line:
                fids.append(line.split()[0])
    return fids


def _clean_cohort_npz(src, is_binary, tmp_dir):
    """Impute y_original NaN (continuous: mean of valid; binary: 0) -> temp NPZ path.

    fit_joint_lasso reads y_original for binary PRS; NaN there crashes the fit.
    """
    src = Path(src)
    z = dict(np.load(src, allow_pickle=True))
    yo = z["y_original"].astype(np.float64)
    nnan = int(np.isnan(yo).sum())
    if nnan:
        fill = 0.0 if is_binary else float(np.nanmean(yo))
        z["y_original"] = np.where(np.isfinite(yo), yo, fill).astype(np.float32)
        tmp = Path(tmp_dir) / f"_tmp_clean_{src.stem}.npz"
        np.savez(tmp, **z)
        return tmp
    return src


# --------------------------------------------------------------------------- building blocks
def prs_load_phenotype(
    ref_npz: Union[str, Path],
    master_phe: Union[str, Path],
    phenotype: str,
    *,
    covariates: Optional[Sequence[str]] = None,
    binary: bool = False,
):
    """Load a phenotype + covariates from master.phe in the reference cohort's sample order.

    Aligns master.phe rows to ``ref_npz['fid_order']`` (= the row order of the scoring cugen), so the
    returned arrays are directly indexable by the split index arrays (``idx_train/val/test``).

    Binary: recode UKBB 1=control/2=case/-9=missing -> 0/1/NaN. Continuous: -9 -> NaN.

    Returns ``(y, Xcov, fids)``: ``y`` (n_samples,), ``Xcov`` (n_samples, n_covar), ``fids`` (str array).
    """
    covariates = list(covariates) if covariates is not None else list(DEFAULT_COVARS)
    z = np.load(ref_npz, allow_pickle=True)
    ref_fids = np.asarray([str(f) for f in z["fid_order"]])
    want = set(["FID"] + covariates + [phenotype])
    phe = (pd.read_csv(master_phe, sep="\t", usecols=lambda c: c in want, dtype={"FID": str})
           .set_index("FID").reindex(ref_fids))
    Xcov = phe[covariates].to_numpy(dtype=np.float64)
    yraw = phe[phenotype].to_numpy(dtype=np.float64)
    if binary:
        y = np.where(yraw == 2, 1.0, np.where(yraw == 1, 0.0, np.nan))
    else:
        y = yraw.copy()
        y[y == -9] = np.nan
    return y, Xcov, ref_fids


def prs_screen(
    cugen_dir: Union[str, Path],
    cohort_npz: Union[str, Path],
    *,
    step2_alpha: float = DEFAULT_STEP2_ALPHA_CONT,
    half_split_chrs: Optional[Sequence[int]] = None,
    max_snps: Optional[int] = None,
    cache_path: Optional[Union[str, Path]] = None,
    device: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Genome-wide step1+2 screen (all 22 chromosomes) -> candidate SNP DataFrame.

    Runs :func:`cugen.screen_chromosome` per chromosome (chr 1-11 split in half at the call level,
    matching production, so a 408k-sample chr 1 fits in GPU memory), concatenates the candidates,
    optionally caps to the top ``max_snps`` by ``strength`` (GPU-mem backstop for the joint fit),
    and optionally caches the result to a feather.

    Parameters
    ----------
    cugen_dir
        Directory of ``chr{1..22}.cugen`` for the cohort being screened (e.g. the train subset).
    cohort_npz
        Residualized cohort NPZ for that split.
    step2_alpha
        Screen FISTA L1 penalty (continuous default 6e-4; binary needs a calibrated smaller value).
    half_split_chrs
        Chromosomes to split into two screen calls (default 1..11).
    max_snps
        If set and exceeded, keep the top ``max_snps`` candidates by ``strength``.
    cache_path
        If set, load from / save to this feather.

    Returns
    -------
    DataFrame of candidates (columns from ``screen_chromosome`` plus ``chr_num``).
    """
    half = set(range(1, 12)) if half_split_chrs is None else set(half_split_chrs)
    if cache_path is not None:
        cache_path = Path(cache_path)
        if cache_path.exists():
            df = pd.read_feather(cache_path)
            if verbose:
                print(f"[prs_screen] loaded cached candidates: {len(df):,} ({cache_path.name})",
                      flush=True)
            return _cap_by_strength(df, max_snps, verbose)

    from .screen import screen_chromosome  # lazy (pulls CuPy)

    cugen_dir = Path(cugen_dir)
    cands = []
    for chrom in range(1, 23):
        cpath = cugen_dir / f"chr{chrom}.cugen"
        nvar = _cugen_nvariants(cpath)
        spans = ([(0, nvar // 2, "_1"), (nvar // 2, nvar, "_2")]
                 if chrom in half else [(0, nvar, "")])
        for vs, ve, suf in spans:
            df = screen_chromosome(chrom, str(cohort_npz), str(cpath),
                                   alpha=step2_alpha, variant_start=vs, variant_end=ve,
                                   output_suffix=suf, device=device)
            if df is not None and len(df):
                df = df.copy()
                df["chr_num"] = chrom
                cands.append(df)
    out = pd.concat(cands, ignore_index=True)
    if verbose:
        print(f"[prs_screen] total candidates: {len(out):,}", flush=True)
    if cache_path is not None:
        out.to_feather(cache_path)
    return _cap_by_strength(out, max_snps, verbose)


def _cap_by_strength(df, max_snps, verbose):
    if max_snps and len(df) > max_snps and "strength" in df.columns:
        df = df.nlargest(max_snps, "strength").reset_index(drop=True)
        if verbose:
            print(f"[prs_screen] capped to top {max_snps:,} by strength (GPU mem)", flush=True)
    return df


def prs_fit_weights(
    candidates,
    cohort_npz: Union[str, Path],
    cugen_dir: Union[str, Path],
    alpha: float,
    *,
    covariates: Optional[Sequence[str]] = None,
    ridge: float = 1e-2,
    loco_mode: str = "lasso",
    phe_file: Optional[Union[str, Path]] = None,
    device: int = 0,
    verbose: bool = False,
) -> pd.DataFrame:
    """Fit the joint LASSO (step 3) at one ``alpha`` and return PRS weights.

    Thin wrapper over :func:`cugen.fit_joint_lasso` that extracts the active set's coefficients.

    Returns
    -------
    DataFrame with ``gidx`` (int64) and ``beta`` (float64) for the active variants
    (empty if the alpha shrinks everything to zero).
    """
    from .lasso import fit_joint_lasso  # lazy (pulls CuPy)

    covariates = list(covariates) if covariates is not None else list(DEFAULT_COVARS)
    res = fit_joint_lasso(candidates, str(cohort_npz), str(cugen_dir),
                          covariates=covariates, alpha=alpha, ridge=ridge, loco_mode=loco_mode,
                          phe_file=(str(phe_file) if phe_file else None),
                          device=device, verbose=verbose)
    aset = res["active_set"]
    act = aset[aset["step3_active"]].copy()
    return pd.DataFrame({"gidx": act["gidx"].astype(np.int64),
                         "beta": act["step3_coef"].astype(np.float64)})


def prs_score(weights, score_cugen_dir, *, device: int = 0, verbose: bool = False) -> np.ndarray:
    """Score every sample in ``score_cugen_dir`` with the given weights -> PRS array.

    Thin wrapper over :func:`cugen.score`; returns a 1-D array ordered by ``sample_idx`` (0-based
    row order of the scoring cugen), so it can be indexed by the split index arrays.
    """
    from .score import score  # lazy (pulls CuPy)

    prs_df = score(weights, str(score_cugen_dir), center=True, device=device, verbose=verbose)
    return prs_df.sort_values("sample_idx")["prs"].to_numpy()


def prs_r2(y, Xcov, prs):
    """Full-model R-squared for a continuous trait: y ~ [1, covariates (+ PRS)].

    Returns ``(r2_cov, r2_full, r2_incremental, n_used)``.
    """
    y = np.asarray(y, dtype=np.float64)
    Xcov = np.asarray(Xcov, dtype=np.float64)
    prs = np.asarray(prs, dtype=np.float64)
    mask = np.isfinite(y) & np.all(np.isfinite(Xcov), axis=1) & np.isfinite(prs)
    y, Xc, p = y[mask], Xcov[mask], prs[mask]
    ones = np.ones((mask.sum(), 1))

    def r2(X):
        c, *_ = np.linalg.lstsq(X, y, rcond=None)
        return 1.0 - np.sum((y - X @ c) ** 2) / np.sum((y - y.mean()) ** 2)

    r2c = r2(np.hstack([ones, Xc]))
    r2f = r2(np.hstack([ones, Xc, p[:, None]]))
    return r2c, r2f, r2f - r2c, int(mask.sum())


def prs_auc(y01, Xcov, prs):
    """Full-model logistic AUC for a binary trait: y ~ covariates (+ PRS), in-sample.

    Returns ``(auc_cov, auc_full, auc_incremental, n_used)``.
    """
    from sklearn.linear_model import LogisticRegression  # lazy
    from sklearn.metrics import roc_auc_score

    y01 = np.asarray(y01, dtype=np.float64)
    Xcov = np.asarray(Xcov, dtype=np.float64)
    prs = np.asarray(prs, dtype=np.float64)
    mask = np.isfinite(y01) & np.all(np.isfinite(Xcov), axis=1) & np.isfinite(prs)
    y, Xc, p = y01[mask].astype(int), Xcov[mask], prs[mask]

    def auc(X):
        Xs = (X - X.mean(0)) / (X.std(0) + 1e-9)
        lr = LogisticRegression(max_iter=2000, C=1e6)
        lr.fit(Xs, y)
        return roc_auc_score(y, lr.predict_proba(Xs)[:, 1])

    a_cov = auc(Xc)
    a_full = auc(np.hstack([Xc, p[:, None]]))
    return a_cov, a_full, a_full - a_cov, int(mask.sum())


def prs_alpha_sweep(
    candidates,
    cohort_npz: Union[str, Path],
    cugen_dir: Union[str, Path],
    score_cugen_dir: Union[str, Path],
    *,
    alphas: Sequence[float],
    y,
    Xcov,
    idx_val,
    idx_train=None,
    binary: bool = False,
    covariates: Optional[Sequence[str]] = None,
    ridge: float = 1e-2,
    loco_mode: str = "lasso",
    phe_file: Optional[Union[str, Path]] = None,
    resume_path: Optional[Union[str, Path]] = None,
    device: int = 0,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sweep step-3 ``alphas``, fitting on ``cohort_npz`` and evaluating each on the validation set.

    For each alpha: fit weights (:func:`prs_fit_weights`), score all samples (:func:`prs_score`),
    and evaluate the full-model metric on ``idx_val`` (and ``idx_train`` if given). Alphas that
    shrink to zero active SNPs are skipped. An optional ``resume_path`` TSV checkpoints each alpha
    so a preempted/requeued run resumes instead of recomputing.

    Parameters
    ----------
    candidates
        Screened candidates (from :func:`prs_screen`) for the fitting cohort.
    cohort_npz, cugen_dir
        Fitting cohort NPZ + its subset cugen dir.
    score_cugen_dir
        Full-cohort cugen used to score ALL samples; ``y``/``Xcov``/``idx_*`` index into its order.
    alphas
        Step-3 joint-LASSO penalties to try.
    y, Xcov
        Full-cohort phenotype (NaN missing) and covariate matrix, aligned to the scoring order.
    idx_val, idx_train
        Index arrays into the full-cohort order for the validation (and optional in-sample) metric.
    binary
        Use logistic AUC (:func:`prs_auc`) instead of R-squared (:func:`prs_r2`).

    Returns
    -------
    DataFrame with one row per evaluated alpha:
    ``alpha, n_active, insample_full, insample_inc, val_cov, val_full, val_inc``.
    """
    evalfn = prs_auc if binary else prs_r2
    metric = "AUC" if binary else "R2"
    y = np.asarray(y)
    Xcov = np.asarray(Xcov)

    rows = []
    done = set()
    if resume_path is not None:
        resume_path = Path(resume_path)
        if resume_path.exists():
            rows = pd.read_csv(resume_path, sep="\t").to_dict("records")
            done = {round(r["alpha"], 12) for r in rows}
            if verbose:
                print(f"[prs_alpha_sweep] resumed {len(rows)} alpha(s) from {resume_path.name}",
                      flush=True)

    for a in alphas:
        a = float(a)
        if round(a, 12) in done:
            if verbose:
                print(f"[prs_alpha_sweep] alpha={a:.1e} already done (resumed)", flush=True)
            continue
        t1 = time.time()
        w = prs_fit_weights(candidates, cohort_npz, cugen_dir, a, covariates=covariates,
                            ridge=ridge, loco_mode=loco_mode, phe_file=phe_file,
                            device=device, verbose=False)
        if len(w) == 0:
            if verbose:
                print(f"[prs_alpha_sweep] alpha={a:.1e} 0 active -- skipped", flush=True)
            continue
        prs_all = prs_score(w, score_cugen_dir, device=device, verbose=False)
        mc_v, mf_v, inc_v, _ = evalfn(y[idx_val], Xcov[idx_val], prs_all[idx_val])
        if idx_train is not None:
            mc_t, mf_t, inc_t, _ = evalfn(y[idx_train], Xcov[idx_train], prs_all[idx_train])
        else:
            mf_t = inc_t = np.nan
        rows.append(dict(alpha=a, n_active=len(w), insample_full=mf_t, insample_inc=inc_t,
                         val_cov=mc_v, val_full=mf_v, val_inc=inc_v))
        if resume_path is not None:
            pd.DataFrame(rows).to_csv(resume_path, sep="\t", index=False)
        if verbose:
            print(f"[prs_alpha_sweep] alpha={a:.1e} n_active={len(w):,} "
                  f"train_{metric}={mf_t:.4f} val_{metric}={mf_v:.4f} "
                  f"val_inc={inc_v:.4f} ({time.time()-t1:.1f}s)", flush=True)

    return pd.DataFrame(rows)


def select_best_alpha(sweep_df: pd.DataFrame, by: str = "val_full") -> float:
    """Return the alpha maximizing column ``by`` (default validation full-model metric)."""
    if sweep_df.empty:
        raise RuntimeError("empty sweep: all alphas gave 0 active SNPs; lower the alpha grid.")
    return float(sweep_df.loc[sweep_df[by].idxmax(), "alpha"])


# --------------------------------------------------------------------------- public pipeline
def build_prs_splits(
    *,
    ref_npz: Union[str, Path],
    master_phe: Union[str, Path],
    source_cugen_dir: Union[str, Path],
    phenotypes: Sequence,
    splits: dict,
    out_dir: Union[str, Path],
    scratch_dir: Union[str, Path],
    covariates: Optional[Sequence[str]] = None,
    subset_splits: Sequence[str] = ("train", "trainval"),
    chunk_size: int = 4096,
    device: int = 0,
    verbose: bool = True,
) -> dict:
    """Build train/val/trainval/test cohorts + subset cugens for the PRS workflow.

    Maps each split's FIDs onto the rows of the reference (full) cohort cugen
    (``cugen row i <-> ref_npz['fid_order'][i]``), residualizes each phenotype per split, and
    materializes subset cugens (with recomputed per-variant stats) for the fitting splits.

    Parameters
    ----------
    ref_npz
        Reference cohort NPZ (the full cohort the source cugen was built from). Must contain
        ``fid_order``, ``sample_idx_sorted``, ``raw_sample_ct``.
    master_phe
        Tab-separated master phenotype file with an ``FID`` column, the covariates, and the
        phenotype columns.
    source_cugen_dir
        Directory of full-cohort ``chr{1..22}.cugen`` files to subset from.
    phenotypes
        Iterable of ``(name, is_binary)`` pairs (or a ``{name: is_binary}`` dict).
    splits
        ``{"train": x, "val": x, "trainval": x, "test": x}`` where each value is either a path to a
        ``.keep`` FID list or a 1-D array of 0-based reference-cohort row indices.
    out_dir
        Writes ``splits/idx_<split>.npy``, ``splits/coverage.txt`` and
        ``cohorts/residual_cohort_<PHENO>_<split>_unified.npz``.
    scratch_dir
        Parent dir for subset cugens: ``<scratch_dir>/prs_cugen_<split>/``.
    subset_splits
        Which splits get residual NPZs + subset cugens built (default train + trainval).
    chunk_size
        Subset batch size; 4096 avoids the int32 overflow (chunk*n_samples > 2^31) at >262k samples.

    Returns
    -------
    dict with ``idx`` (split -> row array), ``coverage`` (DataFrame), ``cohort_npz``
    ({(pheno, split): path}), and ``cugen_dir`` ({split: path}).
    """
    from .subset import subset_cugen_dir  # lazy (pulls CuPy)

    covariates = list(covariates) if covariates is not None else list(DEFAULT_COVARS)
    if isinstance(phenotypes, dict):
        phenotypes = list(phenotypes.items())
    out_dir = Path(out_dir)
    splits_dir = out_dir / "splits"
    cohorts_dir = out_dir / "cohorts"
    splits_dir.mkdir(parents=True, exist_ok=True)
    cohorts_dir.mkdir(parents=True, exist_ok=True)
    scratch_dir = Path(scratch_dir)

    z = np.load(ref_npz, allow_pickle=True)
    fid_order = np.asarray([str(f) for f in z["fid_order"]])
    sample_idx_sorted = np.asarray(z["sample_idx_sorted"])
    raw_sample_ct = int(z["raw_sample_ct"])
    fid_to_row = {f: i for i, f in enumerate(fid_order)}
    if verbose:
        print(f"[prs.splits] reference cohort rows: {len(fid_order):,}", flush=True)

    # 1. split -> sorted reference-cohort rows
    idx = {}
    cov_rows = []
    for split, val in splits.items():
        if isinstance(val, (str, Path)):
            fids = _read_keep_fids(val)
            rows = np.sort(np.asarray([fid_to_row[f] for f in fids if f in fid_to_row],
                                      dtype=np.int64))
            nominal = len(fids)
        else:
            rows = np.sort(np.asarray(val, dtype=np.int64))
            nominal = len(rows)
        idx[split] = rows
        pct = 100.0 * len(rows) / nominal if nominal else float("nan")
        cov_rows.append(dict(split=split, nominal=nominal, in_cohort=len(rows), pct=pct))
        np.save(splits_dir / f"idx_{split}.npy", rows)
        if verbose:
            print(f"[prs.splits] {split}: nominal {nominal:,} -> in-cohort {len(rows):,} "
                  f"({pct:.2f}%)", flush=True)
    coverage = pd.DataFrame(cov_rows)
    (splits_dir / "coverage.txt").write_text(coverage.to_string(index=False) + "\n")

    # disjointness sanity (only when both members present)
    if "train" in idx and "val" in idx:
        assert len(np.intersect1d(idx["train"], idx["val"])) == 0, "train/val overlap!"
    if "trainval" in idx and "test" in idx:
        assert len(np.intersect1d(idx["trainval"], idx["test"])) == 0, "trainval/test overlap!"

    # 2. master phenotypes
    want = set(["FID"] + covariates + [p for p, _ in phenotypes])
    phe = pd.read_csv(master_phe, sep="\t", usecols=lambda c: c in want,
                      dtype={"FID": str}).set_index("FID")

    # 3. residual NPZs for the fitting splits x phenos
    cohort_npz_paths = {}
    for split in subset_splits:
        rows = idx[split]
        split_fids = fid_order[rows]
        Xcov = phe.loc[split_fids, covariates].to_numpy(dtype=np.float64)
        sis_sub = sample_idx_sorted[rows]
        for pheno, is_bin in phenotypes:
            yvals = phe.loc[split_fids, pheno].to_numpy(dtype=np.float64)
            resid, yhat, y_used, mask, coef, r2 = prs_residualize(yvals, Xcov, is_bin)
            outp = cohorts_dir / f"residual_cohort_{pheno}_{split}_unified.npz"
            np.savez(
                outp,
                sample_idx_sorted=sis_sub.astype(np.uint32),
                unsort_idx=np.arange(len(rows), dtype=np.int64),
                y_train=np.nan_to_num(resid, nan=0.0).astype(np.float32),
                y_resid_mask=mask,
                y_original=y_used.astype(np.float32),
                y_hat_covariates=np.nan_to_num(yhat, nan=0.0).astype(np.float32),
                raw_sample_ct=np.int32(raw_sample_ct),
                fid_order=split_fids.astype(object),
                covar_names=np.array(covariates, dtype=object),
                covar_coef=coef.astype(np.float32),
                covar_intercept=np.float32(coef[0]),
                r2_covariates=np.float32(r2),
                phenotype_name=pheno,
                phenotype_is_binary=is_bin,
                cugen_rows=rows.astype(np.int64),
            )
            cohort_npz_paths[(pheno, split)] = outp
            if verbose:
                print(f"[prs.npz] {pheno:9s} {split:8s} N={len(rows):,} "
                      f"valid={int(mask.sum()):,} R2_cov={r2:.4f} -> {outp.name}", flush=True)

    # 4. subset cugens for the fitting splits (GPU)
    cugen_dirs = {}
    for split in subset_splits:
        out_cugen = scratch_dir / f"prs_cugen_{split}"
        out_cugen.mkdir(parents=True, exist_ok=True)
        if verbose:
            print(f"[prs.cugen] subsetting -> {out_cugen} ({len(idx[split]):,} samples)", flush=True)
        subset_cugen_dir(str(source_cugen_dir), str(out_cugen),
                         idx[split].astype(np.int64),
                         device=device, chunk_size=chunk_size, verbose=verbose)
        cugen_dirs[split] = out_cugen

    if verbose:
        print("[prs.splits] done.", flush=True)
    return dict(idx=idx, coverage=coverage, cohort_npz=cohort_npz_paths, cugen_dir=cugen_dirs)


def prs(
    phenotype: str,
    *,
    binary: bool = False,
    splits_dir: Union[str, Path],
    cohorts_dir: Union[str, Path],
    train_cugen_dir: Union[str, Path],
    trainval_cugen_dir: Union[str, Path],
    score_cugen_dir: Union[str, Path],
    ref_npz: Union[str, Path],
    master_phe: Union[str, Path],
    covariates: Optional[Sequence[str]] = None,
    step2_alpha: Optional[float] = None,
    step3_alphas: Optional[Sequence[float]] = None,
    max_step3_snps: int = DEFAULT_MAX_STEP3_SNPS,
    ridge: float = 1e-2,
    loco_mode: str = "lasso",
    half_split_chrs: Optional[Sequence[int]] = None,
    cohort_suffix: str = "unified",
    out_dir: Optional[Union[str, Path]] = None,
    cache_dir: Optional[Union[str, Path]] = None,
    cache: bool = True,
    resume: bool = True,
    device: int = 0,
    verbose: bool = True,
) -> dict:
    """Run the full PRS workflow: screen TRAIN -> sweep step-3 alpha -> select on VAL -> refit
    TRAIN+VAL -> score held-out TEST.

    This is the convenience composition of the building blocks (:func:`prs_screen`,
    :func:`prs_alpha_sweep`, :func:`select_best_alpha`, :func:`prs_fit_weights`, :func:`prs_score`,
    :func:`prs_r2`/:func:`prs_auc`); use those directly to script custom variants.

    Expects the cohorts/splits/subset-cugens produced by :func:`build_prs_splits`:
    ``splits_dir/idx_{train,val,test}.npy`` and
    ``cohorts_dir/residual_cohort_{phenotype}_{train,trainval}_{cohort_suffix}.npz``.

    Selection maximizes the validation full-model metric (R-squared continuous; logistic AUC
    binary). The best alpha is refit on TRAIN+VAL and scored on the held-out TEST split.

    Parameters
    ----------
    phenotype
        Phenotype column name (e.g. ``"INI50"``).
    binary
        ``True`` -> logistic-AUC metric on the LPM (0/1) joint fit; ``False`` -> R-squared.
    score_cugen_dir
        Full-cohort cugen used to score ALL samples (indexed by the split idx arrays).
    step2_alpha
        Screen (step-2) FISTA alpha; defaults to 6e-4 for continuous. Binary traits must pass a
        calibrated value.
    step3_alphas
        Step-3 joint-LASSO alpha grid; defaults to the continuous/binary grid by ``binary``
        (replaces the old ``CG_PRS_ALPHAS`` env override — pass the grid explicitly).
    out_dir
        If set, the per-alpha resume checkpoint, sweep TSV, summary table, figure, and summary text
        are written here. Defaults to ``splits_dir``'s parent.
    cache_dir
        Where candidate feathers live (reuse a prior run's screens while writing fresh outputs to a
        separate ``out_dir``). Defaults to ``out_dir``.

    Returns
    -------
    dict with ``best_alpha``, ``n_variants``, ``insample_cov/insample_full/insample_inc``,
    ``test_cov/test_full/test_inc``, ``gap``, ``sweep`` (DataFrame), and ``weights`` (DataFrame).
    """
    covariates = list(covariates) if covariates is not None else list(DEFAULT_COVARS)
    if step2_alpha is None:
        if binary:
            raise ValueError("binary=True requires an explicit calibrated step2_alpha "
                             "(no safe universal default; e.g. ~5e-7 asthma, ~9e-8 CHD).")
        step2_alpha = DEFAULT_STEP2_ALPHA_CONT
    if step3_alphas is None:
        step3_alphas = (DEFAULT_STEP3_ALPHAS_BIN if binary else DEFAULT_STEP3_ALPHAS_CONT)
    half_split_chrs = (set(range(1, 12)) if half_split_chrs is None else set(half_split_chrs))

    splits_dir = Path(splits_dir)
    cohorts_dir = Path(cohorts_dir)
    out_dir = Path(out_dir) if out_dir is not None else splits_dir.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(cache_dir) if cache_dir is not None else out_dir

    metric = "AUC" if binary else "R2"
    if verbose:
        print(f"[prs] {phenotype} binary={binary} metric={metric} step2_alpha={step2_alpha:.0e} "
              f"step3_grid={list(step3_alphas)}", flush=True)

    # phenotype + covariates in full-cohort order
    y_all, Xcov_all, _ = prs_load_phenotype(ref_npz, master_phe, phenotype,
                                            covariates=covariates, binary=binary)

    idx_train = np.load(splits_dir / "idx_train.npy")
    idx_val = np.load(splits_dir / "idx_val.npy")
    idx_test = np.load(splits_dir / "idx_test.npy")
    idx_tv = np.union1d(idx_train, idx_val)

    suf = cohort_suffix
    train_npz = _clean_cohort_npz(
        cohorts_dir / f"residual_cohort_{phenotype}_train_{suf}.npz", binary, out_dir)
    trainval_npz = _clean_cohort_npz(
        cohorts_dir / f"residual_cohort_{phenotype}_trainval_{suf}.npz", binary, out_dir)

    # screen TRAIN (cached candidates) + alpha sweep on TRAIN, select on VAL
    cand_train = prs_screen(
        train_cugen_dir, train_npz, step2_alpha=step2_alpha, half_split_chrs=half_split_chrs,
        max_snps=max_step3_snps,
        cache_path=(cache_dir / f"_cand_train_{phenotype}_a{step2_alpha:.0e}.feather"
                    if cache else None),
        device=device, verbose=verbose)
    sweep_df = prs_alpha_sweep(
        cand_train, train_npz, train_cugen_dir, score_cugen_dir,
        alphas=step3_alphas, y=y_all, Xcov=Xcov_all, idx_val=idx_val, idx_train=idx_train,
        binary=binary, covariates=covariates, ridge=ridge, loco_mode=loco_mode,
        phe_file=master_phe,
        resume_path=(out_dir / f"_sweep_resume_{phenotype}_a{step2_alpha:.0e}.tsv"
                     if resume else None),
        device=device, verbose=verbose)
    if sweep_df.empty:
        raise RuntimeError(f"all alphas gave 0 active SNPs for {phenotype}; lower step3_alphas.")
    sweep_df.to_csv(out_dir / f"{phenotype}_prs_sweep.tsv", sep="\t", index=False)
    best_alpha = select_best_alpha(sweep_df)
    if verbose:
        print(f"[prs.select] best alpha={best_alpha:.1e}", flush=True)

    # refit best on TRAIN+VAL, score held-out TEST
    cand_tv = prs_screen(
        trainval_cugen_dir, trainval_npz, step2_alpha=step2_alpha, half_split_chrs=half_split_chrs,
        max_snps=max_step3_snps,
        cache_path=(cache_dir / f"_cand_trainval_{phenotype}_a{step2_alpha:.0e}.feather"
                    if cache else None),
        device=device, verbose=verbose)
    w_final = prs_fit_weights(cand_tv, trainval_npz, trainval_cugen_dir, best_alpha,
                              covariates=covariates, ridge=ridge, loco_mode=loco_mode,
                              phe_file=master_phe, device=device, verbose=False)
    prs_all = prs_score(w_final, score_cugen_dir, device=device, verbose=False)
    evalfn = prs_auc if binary else prs_r2
    mc_is, mf_is, inc_is, n_is = evalfn(y_all[idx_tv], Xcov_all[idx_tv], prs_all[idx_tv])
    mc_te, mf_te, inc_te, n_te = evalfn(y_all[idx_test], Xcov_all[idx_test], prs_all[idx_test])
    gap = mf_is - mf_te

    result = dict(
        phenotype=phenotype, metric=metric, binary=binary, best_alpha=best_alpha,
        n_variants=len(w_final),
        insample_cov=mc_is, insample_full=mf_is, insample_inc=inc_is, n_insample=n_is,
        test_cov=mc_te, test_full=mf_te, test_inc=inc_te, n_test=n_te,
        gap=gap, sweep=sweep_df, weights=w_final)

    if verbose:
        print(f"\n[prs] {phenotype}: best_alpha={best_alpha:.1e} n_active={len(w_final):,} "
              f"in-sample {metric}={mf_is:.4f} TEST {metric}={mf_te:.4f} gap={gap:.4f}", flush=True)

    if out_dir is not None:
        _write_prs_outputs(result, out_dir, covariates)
    return result


def _write_prs_outputs(result, out_dir, covariates):
    """Write table_prs_<PHENO>.csv, fig_prs_<PHENO>.{pdf,png}, <PHENO>_prs.summary.txt."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p = result["phenotype"]
    m = result["metric"]
    out_dir = Path(out_dir)

    table = pd.DataFrame([dict(
        method="UltraLasso (uniLasso: LOO + non-neg lasso select + joint refit)",
        phenotype=p, metric=m, best_alpha=result["best_alpha"], n_variants=result["n_variants"],
        insample_cov=result["insample_cov"], insample_full=result["insample_full"],
        insample_inc=result["insample_inc"], n_insample=result["n_insample"],
        test_cov=result["test_cov"], test_full=result["test_full"],
        test_inc=result["test_inc"], n_test=result["n_test"],
        gap_insample_minus_test=result["gap"])])
    table.to_csv(out_dir / f"table_prs_{p}.csv", index=False)

    with open(out_dir / f"{p}_prs.summary.txt", "w") as fh:
        fh.write(f"UltraLasso PRS {p} ({m}); manuscript split, array, unified\n\n")
        fh.write(f"best step3 alpha      : {result['best_alpha']:.1e}\n")
        fh.write(f"final active variants : {result['n_variants']:,}\n")
        fh.write(f"in-sample (train+val) full {m}: {result['insample_full']:.4f} "
                 f"(cov {result['insample_cov']:.4f}, +PRS {result['insample_inc']:.4f})\n")
        fh.write(f"TEST full {m}                 : {result['test_full']:.4f} "
                 f"(cov {result['test_cov']:.4f}, +PRS {result['test_inc']:.4f})\n")
        fh.write(f"gap (in-sample - test)            : {result['gap']:.4f}\n\n")
        fh.write("UltraLasso = uniLasso (step1+2 LOO univariate + non-neg lasso select; "
                 "step3 joint refit); compare to preprint uniLasso, NOT uniLasso-ES.\n\n")
        fh.write(result["sweep"].to_string(index=False))

    fig, ax = plt.subplots(figsize=(4.6, 4.0))
    vals = [result["insample_full"], result["test_full"]]
    bars = ax.bar(["in-sample\n(train+val)", "held-out\ntest"], vals,
                  color=["#0072B2", "#D55E00"], width=0.6, edgecolor="white")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.003, f"{v:.3f}",
                ha="center", va="bottom", fontweight="bold", fontsize=10)
    lo = min(vals) - 0.05 if result["binary"] else 0
    ax.set_ylim(lo, max(vals) * 1.05 + 0.02)
    ax.set_ylabel(f"Full-model {m} (covariates + PRS)")
    ax.set_title(f"UltraLasso {p} PRS (alpha={result['best_alpha']:.0e}, "
                 f"{result['n_variants']:,} SNPs)", fontsize=9.5)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_dir / f"fig_prs_{p}.pdf")
    fig.savefig(out_dir / f"fig_prs_{p}.png", dpi=300)
    plt.close(fig)
