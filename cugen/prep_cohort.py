"""cugen.prep_cohort — build a residualised cohort NPZ.

Given a phenotype + covariate file and a base-cohort reference NPZ (which
defines the canonical sample ordering of the .cugen files), produce an NPZ
compatible with the rest of the cugen pipeline:

    sample_idx_sorted, unsort_idx, y_train, y_original, y_hat_covariates,
    raw_sample_ct, fid_order, covar_*, phenotype_name, phenotype_is_binary,
    base_cohort, subset_rows_in_base_cugen

If the phenotype column looks binary (unique values ⊆ {0, 1, 2, -9}, or
the caller passes ``binary_recode=True``), it is recoded to 0/1 with NaN
for missing. After recode the phenotype is residualised against the
covariates by OLS (a linear probability model for binary, plain OLS for
continuous). The y_train residual is what subsequent steps consume.

The function is dataset-agnostic: it does not assume any specific biobank
or phenotype-naming convention. You provide the master phenotype TSV
path, PSAM path, and reference NPZ explicitly.
"""

import time
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd


def _maybe_recode_binary(y: pd.Series, force_binary: Optional[bool] = None):
    """Detect + recode binary encoding (1=control, 2=case, -9=missing → 0/1/NaN).

    Falls back to no-op when the column doesn't look binary, unless
    ``force_binary=True`` is set.
    """
    u = pd.unique(y.dropna())
    looks_binary = set(u).issubset({1, 2, -9, 0})
    if force_binary is False:
        return y, False
    if not (looks_binary or force_binary):
        return y, False
    y2 = y.copy().replace({-9: np.nan, 1: 0, 2: 1})
    return y2, True


def prepare_cohort(
    phenotype: str,
    ref_npz: Union[str, Path],
    master_phe: Union[str, Path],
    psam: Union[str, Path],
    n_samples: int,
    *,
    output: Optional[Union[str, Path]] = None,
    covariates: Optional[list] = None,
    binary: Optional[bool] = None,
    impute_missing: Optional[str] = None,
    verbose: bool = True,
) -> dict:
    """Build a residualised cohort NPZ.

    Parameters
    ----------
    phenotype : str
        Column name in ``master_phe`` to use as the outcome.
    ref_npz : path
        Reference NPZ for the base cohort. Must contain
        ``sample_idx_sorted`` (strictly increasing pgen indices) and
        ``fid_order`` (parallel FID array). This defines the canonical
        sample ordering of the .cugen files.
    master_phe : path
        Tab-separated phenotype/covariate file. Must have an ``FID`` (or
        ``#FID``) column plus ``phenotype`` and every ``covariates`` column.
    psam : path
        Plink2 PSAM file (tab-separated) mapping FID → row index. Row
        order in the PSAM defines pgen_idx.
    n_samples : int
        Expected sample count in ``ref_npz``. Sanity-check only.
    output : path, optional
        Path to write the .npz to. If None, no file is written (return
        dict only).
    covariates : list of str
        Covariate column names. No default — pass them explicitly.
    binary : bool, optional
        ``True`` to force binary recoding, ``False`` to skip the check,
        ``None`` (default) to auto-detect from value set.
    verbose : bool

    Returns
    -------
    dict
        Mapping with all keys of the saved NPZ. Companion file with the
        suffix ``.subset_indices.npy`` is also written when ``output`` is set.
    """
    if covariates is None:
        raise ValueError(
            "covariates must be passed explicitly — there is no default."
        )
    covariates = list(covariates)
    n_base = int(n_samples)

    def log(msg):
        if verbose:
            print(f"[prep] {msg}", flush=True)

    log(f"phenotype={phenotype}  (N_base={n_base:,})")
    log(f"ref_npz={ref_npz}  covars={covariates}")
    t0 = time.time()

    # --- Base cohort canonical ordering -----------------------------------
    ref = np.load(ref_npz, allow_pickle=True)
    base_sample_idx = np.asarray(ref["sample_idx_sorted"], dtype=np.int64)
    base_fids = np.asarray(ref["fid_order"], dtype=str)
    if len(base_sample_idx) != n_base:
        raise ValueError(
            f"ref NPZ has {len(base_sample_idx)} samples, expected {n_base} "
            f"in ref_npz"
        )
    if not np.all(np.diff(base_sample_idx) > 0):
        raise ValueError("ref NPZ sample_idx_sorted must be strictly increasing")
    log(f"base cohort cugen ordering loaded: {n_base:,} samples")

    pgen_to_base_row = {int(p): i for i, p in enumerate(base_sample_idx)}
    base_fid_to_row = {str(f): i for i, f in enumerate(base_fids)}

    # --- psam: FID → pgen_idx --------------------------------------------
    psam_df = pd.read_csv(psam, sep="\t", dtype=str, low_memory=False)
    psam_df.columns = [c.lstrip("#") for c in psam_df.columns]
    psam_df["pgen_idx"] = np.arange(len(psam_df), dtype=np.int64)
    log(f"pgen: {len(psam_df):,} samples")

    # --- master.phe (FID / #FID auto-detect) -----------------------------
    with open(master_phe, "r") as f:
        header0 = f.readline().rstrip("\n").split("\t")
    fid_col = "#FID" if "#FID" in header0 else "FID"
    cols_needed = [fid_col, phenotype] + covariates
    t = time.time()
    phe = pd.read_csv(
        master_phe, sep="\t", usecols=cols_needed, dtype={fid_col: str},
        low_memory=False, na_values=["NA", "NaN", "nan", ""],
    )
    if fid_col != "FID":
        phe = phe.rename(columns={fid_col: "FID"})
    log(f"read master.phe {len(phe):,} rows in {time.time()-t:.1f}s "
        f"(fid column = {fid_col!r})")

    phe[phenotype], is_binary = _maybe_recode_binary(phe[phenotype], binary)
    # -9 is treated as missing on both binary and continuous columns.
    # _maybe_recode_binary handles -9 only when it recognises a binary column,
    # so apply the same recoding for continuous traits here as well.
    if not is_binary:
        phe[phenotype] = phe[phenotype].replace(-9, np.nan)

    phe_in_base = phe[phe["FID"].isin(base_fid_to_row)].copy()
    if impute_missing:
        # Keep EVERY base sample (drop only covariate-missing rows). Samples
        # with a missing phenotype get a residual of 0 below, so they sit at
        # the cohort mean and contribute nothing to screen/LASSO/GWAS — N then
        # equals the full cugen and no cohort-matched subset cugen is needed.
        phe_clean = phe_in_base.dropna(subset=covariates).copy()
        n_miss = int(phe_clean[phenotype].isna().sum())
        log(f"after intersect (impute_missing): {len(phe_clean):,} "
            f"({len(phe_clean)/n_base*100:.1f}% of base); "
            f"{n_miss:,} phenotype-missing → residual 0")
    else:
        phe_clean = phe_in_base.dropna(subset=[phenotype] + covariates).copy()
        log(f"after intersect + dropna: {len(phe_clean):,} "
            f"({len(phe_clean)/n_base*100:.1f}% of base cohort)")

    merged = psam_df[["FID", "pgen_idx"]].merge(phe_clean, on="FID", how="inner")
    merged = merged.sort_values("pgen_idx", kind="stable").reset_index(drop=True)
    n_out = len(merged)
    log(f"merged with pgen order: {n_out:,} samples")

    sample_idx_sorted = merged["pgen_idx"].to_numpy(dtype=np.uint32)
    fid_order = merged["FID"].to_numpy()
    y_original = merged[phenotype].to_numpy(dtype=np.float32)
    X_cov = merged[covariates].to_numpy(dtype=np.float32)

    # --- Residualise (OLS / LPM) -----------------------------------------
    # Self-contained closed-form OLS (no sklearn dep): β = (XᵀX)⁻¹ Xᵀy
    Xb = np.concatenate([X_cov, np.ones((n_out, 1), dtype=np.float32)], axis=1)
    # Fit covariate model on phenotype-present rows only; impute-missing rows
    # (NaN phenotype) are excluded from the fit and given residual 0.
    fit_mask = ~np.isnan(y_original)
    coef_full, *_ = np.linalg.lstsq(Xb[fit_mask], y_original[fit_mask], rcond=None)
    intercept = float(coef_full[-1])
    coef = coef_full[:-1].astype(np.float32)
    y_hat = (X_cov @ coef + intercept).astype(np.float32)
    y_train = (y_original - y_hat).astype(np.float32)
    if not fit_mask.all():
        # neutral imputation: residual 0, y_original set to the covariate fit
        y_train[~fit_mask] = 0.0
        y_original[~fit_mask] = y_hat[~fit_mask]
    ss_res = float(np.sum((y_original[fit_mask] - y_hat[fit_mask]) ** 2))
    ss_tot = float(np.sum((y_original[fit_mask] - y_original[fit_mask].mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
    log(f"covariate R² = {r2:.4f}   y_train: mean={y_train.mean():.3g} "
        f"std={y_train.std():.3g}")

    mu = X_cov.mean(axis=0).astype(np.float32)
    sd = X_cov.std(axis=0).astype(np.float32)
    sd = np.where(sd > 0, sd, 1.0).astype(np.float32)

    unsort_idx = np.argsort(np.argsort(sample_idx_sorted)).astype(np.int64)

    subset_rows = np.asarray(
        [pgen_to_base_row[int(p)] for p in sample_idx_sorted], dtype=np.int64
    )
    if not np.all((subset_rows >= 0) & (subset_rows < n_base)):
        raise RuntimeError("subset_rows out of base cohort range")
    log(f"subset positions in base cugen: min={subset_rows.min()} "
        f"max={subset_rows.max()} n={len(subset_rows):,}")

    out_kw = dict(
        sample_idx_sorted=sample_idx_sorted,
        unsort_idx=unsort_idx,
        y_train=y_train,
        y_original=y_original,
        y_hat_covariates=y_hat,
        raw_sample_ct=len(psam_df),
        fid_order=fid_order,
        covar_names=np.asarray(covariates),
        covar_coef=coef,
        covar_intercept=np.float32(intercept),
        r2_covariates=np.float32(r2),
        covar_mu=mu,
        covar_sd=sd,
        phenotype_name=phenotype,
        phenotype_is_binary=bool(is_binary),
        base_cohort=str(ref_npz),
        subset_rows_in_base_cugen=subset_rows,
    )

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.savez(out, **out_kw)
        log(f"wrote {out}  ({n_out:,} samples, {time.time()-t0:.1f}s)")
        idx_npy = out.with_suffix(".subset_indices.npy")
        np.save(idx_npy, subset_rows)
        log(f"wrote companion {idx_npy}")

    return out_kw
