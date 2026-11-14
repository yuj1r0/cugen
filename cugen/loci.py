"""cugen.loci - Step 5a: define GWAS-significant loci from sumstats.

Verbatim port of ``step5a_define_loci.py`` (production).

Algorithm:
  1. Filter to genome-wide significant (P < p_threshold)
  2. Greedy clump: pick top SNP, remove all within +/- clump_window, repeat
  3. Extend each lead SNP +/- flank to define locus boundaries
  4. Merge overlapping loci (transitive closure when merge_overlapping=True)
  5. Special handling for HLA region (chr6: 25-34 Mb) - merge all HLA-overlapping
     loci into one.

Output columns (production + spec aliases):
  chrom, start, end, lead_pos, lead_p, n_signif, span_kb,
  locus_id, CHR, start_bp, end_bp, size_kb, lead_gidx, lead_id,
  n_gws_variants, is_hla
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger("cugen.loci")

#: median of the chi-square distribution with 1 df (scipy.stats.chi2.ppf(0.5, 1))
CHI2_MEDIAN = 0.454936423119572
#: |z| at two-sided P = 5e-8 (scipy.stats.norm.isf(2.5e-8))
ZGWS = 5.451310437845478


def lambda_gc(z=None, p=None) -> float:
    """Genomic-control lambda: median chi-square / its null median (``CHI2_MEDIAN``).

    Pass Z statistics (``z=``, chi2 = z^2, no special functions) or P values (``p=``,
    clipped to [1e-300, 1]). For P only the middle order statistic(s) are converted, since
    ``chi2.isf`` is monotone, which avoids an ``isf`` over millions of values. Non-finite
    entries are ignored; NaN when nothing is left.
    """
    if (z is None) == (p is None):
        raise ValueError("pass exactly one of z= or p=")
    if z is not None:
        z = np.asarray(z, dtype=np.float64)
        z = z[np.isfinite(z)]
        return float(np.median(z * z) / CHI2_MEDIAN) if z.size else float("nan")
    p = np.asarray(p, dtype=np.float64)
    p = np.clip(p[np.isfinite(p)], 1e-300, 1.0)
    m = p.size
    if not m:
        return float("nan")
    from scipy.stats import chi2
    k = (m - 1) // 2
    part = np.partition(p, [k, m - 1 - k])
    return float(0.5 * (chi2.isf(part[k], 1) + chi2.isf(part[m - 1 - k], 1)) / CHI2_MEDIAN)


def log10p_from_z(z):
    """-log10 two-sided P from Z, computed in log space (no underflow at large |z|)."""
    from scipy.special import log_ndtr
    z = np.abs(np.asarray(z, dtype=np.float64))
    return -(np.log(2.0) + log_ndtr(-z)) / np.log(10.0)


def _greedy_clump(chrom, pos, order, window):
    """Greedy distance clumping core shared by :func:`define_loci` and :func:`distance_clumps`.

    Variants are visited in ``order`` (priority, best first); a variant becomes a lead
    unless an earlier lead on the same chromosome lies within +/- ``window`` (inclusive).
    Returns (lead indices in visiting order, number of not-yet-claimed variants within
    the window of each lead, itself included). Per chromosome the positions are sorted
    once and each lead claims a searchsorted range, so the cost is O(G log G).
    """
    chrom = np.asarray(chrom)
    pos = pd.to_numeric(pd.Series(np.asarray(pos, dtype=object) if np.asarray(pos).dtype == object
                                  else np.asarray(pos)), errors="coerce").to_numpy(np.float64, na_value=np.nan)
    order = np.asarray(order, dtype=np.int64)
    G = len(pos)
    rank = np.empty(G, dtype=np.int64)
    rank[order] = np.arange(G)
    lead = np.zeros(G, dtype=bool)
    nsig = np.zeros(G, dtype=np.int64)
    nopos = ~np.isfinite(pos)
    lead[nopos] = True                  # no position: its own locus, claims nothing
    nsig[nopos] = 1
    for c in pd.unique(chrom):
        idx = np.flatnonzero((chrom == c) & ~nopos)
        if not len(idx):
            continue
        o = idx[np.argsort(pos[idx], kind="stable")]
        ps = pos[o]
        taken = np.zeros(len(o), dtype=bool)
        for i in np.argsort(rank[o], kind="stable"):
            if taken[i]:
                continue
            lo = np.searchsorted(ps, ps[i] - window, "left")
            hi = np.searchsorted(ps, ps[i] + window, "right")
            lead[o[i]] = True
            nsig[o[i]] = (hi - lo) - int(taken[lo:hi].sum())
            taken[lo:hi] = True
    leads = order[lead[order]]
    return leads, nsig[leads]


def distance_clumps(chrom, pos, score, window=1_000_000):
    """Greedy distance clumps: the best variant (largest ``score``), drop everything within
    +/- ``window``, repeat, per chromosome. Returns a DataFrame CHR, POS, LOG10P (= score),
    n_sig (variants claimed by the lead, itself included) and row (the lead's index in the
    input arrays), sorted by score (descending)."""
    chrom = np.asarray(chrom)
    pos = np.asarray(pos)
    score = np.asarray(score, dtype=np.float64)
    leads, nsig = _greedy_clump(chrom, pos, np.argsort(-score, kind="stable"), window)
    df = pd.DataFrame({"CHR": chrom[leads].astype(np.int64), "POS": pos[leads].astype(np.int64),
                       "LOG10P": score[leads], "n_sig": nsig.astype(np.int64),
                       "row": leads.astype(np.int64)})
    return df.sort_values("LOG10P", ascending=False, kind="stable").reset_index(drop=True)


def summarize(sumstats=None, *, z=None, chrom=None, pos=None, z_col="Z", window=1_000_000):
    """Fast GWAS summary from Z alone (no ``chi2.isf`` over millions of P values).

    Returns ``(info, clumps)``: ``info`` = dict(n_tests, lambda_gc, n_gws, n_loci,
    n_chrom_with_gws, max_abs_z, max_log10p) with GWS = |z| > ``ZGWS`` (P < 5e-8), and
    ``clumps`` = :func:`distance_clumps` over the GWS rows (``window`` bp, default 1 Mb;
    ``row`` indexes the input rows).
    Pass a sumstats DataFrame (CHR, POS and ``z_col``) or the arrays.
    """
    if sumstats is not None:
        z = sumstats[z_col].to_numpy()
        chrom = sumstats["CHR"].to_numpy()
        pos = sumstats["POS"].to_numpy()
    z = np.asarray(z, dtype=np.float64)
    ok = np.isfinite(z)
    g = ok & (np.abs(z) > ZGWS)
    cl = distance_clumps(np.asarray(chrom)[g], np.asarray(pos)[g], log10p_from_z(z[g]), window)
    cl["row"] = np.flatnonzero(g)[cl["row"].to_numpy()]          # index into the input rows
    return {"n_tests": int(ok.sum()), "lambda_gc": lambda_gc(z=z[ok]), "n_gws": int(g.sum()),
            "n_loci": int(len(cl)), "n_chrom_with_gws": int(cl.CHR.nunique()) if len(cl) else 0,
            "max_abs_z": float(np.abs(z[ok]).max()) if ok.any() else float("nan"),
            "max_log10p": float(log10p_from_z(np.abs(z[ok]).max())) if ok.any() else float("nan")}, cl


def define_loci(
    sumstats,
    *,
    p_threshold: float = 5e-8,
    window_kb: int = 500,
    merge_overlapping: bool = True,
    output: Optional[Union[str, Path]] = None,
    clump_window: int = 1_000_000,
    hla_start: int = 25_000_000,
    hla_end: int = 34_000_000,
    chr_filter: Optional[int] = None,
    max_loci: Optional[int] = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """Define genome-wide significant loci from a sumstats table.

    Parameters
    ----------
    sumstats : pandas.DataFrame or path to tsv / tsv.gz
        Must contain at least [CHR, POS, P] (production column names).
        ``gidx`` and ``ID`` are used if present.
    p_threshold : float, default 5e-8
        Significance threshold.
    window_kb : int, default 500
        Flank radius around each lead variant in kilobases. Maps to the
        production ``flank`` parameter (``flank = window_kb * 1000``).
    merge_overlapping : bool, default True
        Transitively merge overlapping windows (step 4). When False, the
        clumped/flanked loci are returned unmerged.
    output : str or Path, optional
        If given, write the resulting loci DataFrame as a TSV to this path.
    clump_window : int, default 1_000_000
        Clumping window in bp (+/-) used in step 2.
    hla_start, hla_end : int
        HLA region boundaries on chr6 (default 25-34 Mb).
    chr_filter : int, optional
        If set, only process this chromosome.
    verbose : bool, default False
        Log per-step progress at INFO level.

    Returns
    -------
    pandas.DataFrame
        Loci with production columns plus spec aliases
        (``chrom, start, end, lead_pos, lead_p, n_signif, span_kb``).
    """
    flank = int(window_kb) * 1000

    # ---- Load -------------------------------------------------------------
    if isinstance(sumstats, (str, Path)):
        path = str(sumstats)
        if verbose:
            logger.info("Loading summary statistics from %s", path)
        df = pd.read_csv(path, sep="\t")
    elif isinstance(sumstats, pd.DataFrame):
        df = sumstats.copy()
    else:
        raise TypeError(
            "sumstats must be a pandas DataFrame or a path to a tsv/tsv.gz file; "
            f"got {type(sumstats).__name__}"
        )

    # Ensure CHR is integer (filter out non-numeric like '.' from imputed data)
    df = df[pd.to_numeric(df["CHR"], errors="coerce").notna()].copy()
    df["CHR"] = df["CHR"].astype(int)

    if chr_filter is not None:
        df = df[df["CHR"] == chr_filter].copy()
        if verbose:
            logger.info("  Filtered to chr%s: %d variants", chr_filter, len(df))

    if verbose:
        logger.info("  Total variants: %d", len(df))

    empty_cols = [
        "locus_id", "CHR", "start_bp", "end_bp", "size_kb",
        "lead_gidx", "lead_pos", "lead_p", "lead_id",
        "n_gws_variants", "is_hla",
    ]

    # ---- Step 1: filter to genome-wide significant -----------------------
    sig = df[df["P"] < p_threshold].copy()
    if verbose:
        logger.info(
            "  Genome-wide significant (P < %s): %d", p_threshold, len(sig)
        )

    if len(sig) == 0:
        if verbose:
            logger.info("  No significant variants found. Returning empty loci.")
        loci_df = pd.DataFrame(columns=empty_cols)
        loci_df = _add_spec_aliases(loci_df)
        if output is not None:
            loci_df.to_csv(output, sep="\t", index=False)
        return loci_df

    # ---- Step 2: greedy clumping -----------------------------------------
    # Visit variants by P; a variant leads unless an earlier lead on its
    # chromosome is within +/- clump_window (inclusive). Shared, vectorised core.
    sig = sig.sort_values("P").reset_index(drop=True)
    lead_idx, _ = _greedy_clump(sig["CHR"].to_numpy(), sig["POS"].to_numpy(),
                                np.arange(len(sig)), clump_window)
    lead_snps = [sig.iloc[i] for i in lead_idx]

    if verbose:
        logger.info(
            "  Lead SNPs after clumping (+/-%.0fMb): %d",
            clump_window / 1e6,
            len(lead_snps),
        )

    # ---- Step 3: extend flanks to define locus boundaries ----------------
    loci = []
    for lead in lead_snps:
        start = max(0, int(lead["POS"]) - flank)
        end = int(lead["POS"]) + flank

        # Count GWS variants in this locus
        n_gws = len(
            sig[
                (sig["CHR"] == lead["CHR"])
                & (sig["POS"] >= start)
                & (sig["POS"] <= end)
            ]
        )

        loci.append(
            {
                "CHR": int(lead["CHR"]),
                "start_bp": start,
                "end_bp": end,
                "lead_gidx": int(lead["gidx"]) if "gidx" in lead.index else -1,
                "lead_pos": int(lead["POS"]),
                "lead_p": float(lead["P"]),
                "lead_id": lead.get("ID", "NA"),
                "n_gws_variants": n_gws,
            }
        )

    loci_df = pd.DataFrame(loci)
    loci_df = loci_df.sort_values(["CHR", "start_bp"]).reset_index(drop=True)

    if verbose:
        logger.info("  Loci before merging: %d", len(loci_df))

    # ---- Step 4: merge overlapping loci ----------------------------------
    if merge_overlapping:
        merged = []
        for _, row in loci_df.iterrows():
            if (
                merged
                and merged[-1]["CHR"] == row["CHR"]
                and row["start_bp"] <= merged[-1]["end_bp"]
            ):
                # Merge: extend end, keep the more significant lead
                prev = merged[-1]
                prev["end_bp"] = max(prev["end_bp"], row["end_bp"])
                prev["n_gws_variants"] += row["n_gws_variants"]
                if row["lead_p"] < prev["lead_p"]:
                    prev["lead_gidx"] = row["lead_gidx"]
                    prev["lead_pos"] = row["lead_pos"]
                    prev["lead_p"] = row["lead_p"]
                    prev["lead_id"] = row["lead_id"]
            else:
                merged.append(dict(row))

        loci_df = pd.DataFrame(merged)

    # ---- Step 5: HLA special handling ------------------------------------
    if not loci_df.empty:
        chr6 = loci_df[loci_df["CHR"] == 6]
        hla_loci = chr6[(chr6["start_bp"] < hla_end) & (chr6["end_bp"] > hla_start)]

        if len(hla_loci) > 1:
            non_hla = loci_df[~loci_df.index.isin(hla_loci.index)]
            hla_merged = {
                "CHR": 6,
                "start_bp": int(hla_loci["start_bp"].min()),
                "end_bp": int(hla_loci["end_bp"].max()),
                "lead_gidx": int(
                    hla_loci.loc[hla_loci["lead_p"].idxmin(), "lead_gidx"]
                ),
                "lead_pos": int(
                    hla_loci.loc[hla_loci["lead_p"].idxmin(), "lead_pos"]
                ),
                "lead_p": float(hla_loci["lead_p"].min()),
                "lead_id": hla_loci.loc[hla_loci["lead_p"].idxmin(), "lead_id"],
                "n_gws_variants": int(hla_loci["n_gws_variants"].sum()),
            }
            loci_df = pd.concat(
                [non_hla, pd.DataFrame([hla_merged])], ignore_index=True
            )
            loci_df = loci_df.sort_values(["CHR", "start_bp"]).reset_index(drop=True)
            if verbose:
                logger.info("  Merged %d HLA loci into 1", len(hla_loci))

    # ---- Locus IDs, sizes, HLA flag --------------------------------------
    loci_df["locus_id"] = range(1, len(loci_df) + 1)
    loci_df["size_kb"] = (loci_df["end_bp"] - loci_df["start_bp"]) / 1000

    loci_df["is_hla"] = (
        (loci_df["CHR"] == 6)
        & (loci_df["start_bp"] < hla_end)
        & (loci_df["end_bp"] > hla_start)
    ).astype(int)

    # Production column order
    cols = [
        "locus_id", "CHR", "start_bp", "end_bp", "size_kb",
        "lead_gidx", "lead_pos", "lead_p", "lead_id",
        "n_gws_variants", "is_hla",
    ]
    loci_df = loci_df[cols]

    # Cap to the top-N most significant loci (smallest lead_p). Used by the
    # "seconds per 1000 loci" benchmark pass. Deterministic mergesort so the
    # cut is reproducible; re-sort to genomic order + renumber locus_id so the
    # capped set stays internally consistent with the full-run contract.
    if max_loci is not None and len(loci_df) > int(max_loci):
        loci_df = (
            loci_df.sort_values("lead_p", kind="mergesort")
            .head(int(max_loci))
            .sort_values(["CHR", "start_bp"])
            .reset_index(drop=True)
        )
        loci_df["locus_id"] = range(1, len(loci_df) + 1)
        if verbose:
            logger.info("  Capped to top %d loci by lead_p", int(max_loci))

    # Add spec-required aliases (chrom, start, end, n_signif, span_kb)
    loci_df = _add_spec_aliases(loci_df)

    if verbose:
        logger.info("  Final loci: %d", len(loci_df))
        if len(loci_df) > 0:
            logger.info(
                "  Total GWS variants covered: %d",
                int(loci_df["n_gws_variants"].sum()),
            )
            logger.info(
                "  Median locus size: %.0f kb", loci_df["size_kb"].median()
            )
            logger.info("  HLA loci: %d", int(loci_df["is_hla"].sum()))

    if output is not None:
        loci_df.to_csv(output, sep="\t", index=False)
        if verbose:
            logger.info("Saved %d loci to %s", len(loci_df), output)

    return loci_df


def _add_spec_aliases(loci_df: pd.DataFrame) -> pd.DataFrame:
    """Add the spec-required alias columns alongside production columns.

    Aliases:
      chrom    <- CHR
      start    <- start_bp
      end      <- end_bp
      n_signif <- n_gws_variants
      span_kb  <- size_kb
    (lead_pos and lead_p already exist in the production schema.)
    """
    if loci_df.empty:
        for c in ("chrom", "start", "end", "n_signif", "span_kb"):
            if c not in loci_df.columns:
                loci_df[c] = pd.Series(dtype="float64")
        return loci_df

    loci_df = loci_df.copy()
    loci_df["chrom"] = loci_df["CHR"]
    loci_df["start"] = loci_df["start_bp"]
    loci_df["end"] = loci_df["end_bp"]
    loci_df["n_signif"] = loci_df["n_gws_variants"]
    loci_df["span_kb"] = loci_df["size_kb"]
    return loci_df


__all__ = ["define_loci"]
