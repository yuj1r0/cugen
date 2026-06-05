"""cugen.plot.qq — GPU-native QQ plot.

CuPy does the heavy lifting (sort, expected quantiles, lambda_GC, MAF
stratification), then matplotlib renders the panel. Square panel with
manuscript-style fixed x-axis cap (CLAUDE.md `feedback_figures_biobank_style`).

Batched inputs are supported in the same `qq()` call:

  * single  — ``qq(sumstats, output)``           → returns one result dict
  * list    — ``qq([df1, df2], [out1, out2])``    → returns list of dicts
  * dict    — ``qq({'height': p1, 'bmi': p2}, output_dir='qc/', ...)``
              → returns ``{label: dict}``; filenames derived as
              ``{output_dir}/qq_{label}.png``.

For batched modes, ``n_workers > 1`` spawns a ``multiprocessing.Pool``
(spawn context) with CUDA MPS-style sharing. Same multiprocess pattern as
``cg.gwas`` — session-14 validated 8 workers on one H100 SXM5.

Returns a dict (or list/dict of dicts) with the computed arrays + lambda_GC
so callers can use this for figures AND LDSC-style summary stats without
re-reading sumstats.
"""

import multiprocessing as mp
import os
from pathlib import Path
from typing import Mapping, Optional, Sequence, Union

import numpy as np
import pandas as pd

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False
    cp = None


def _to_xp(x, xp):
    if xp is cp:
        return cp.asarray(x)
    return np.asarray(x)


def _lambda_gc_from_chisq(chisq_xp, xp):
    """lambda_GC = median(chisq) / 0.4549 (the 1-df chi-sq median)."""
    return float(xp.median(chisq_xp)) / 0.4549


def _qq_quantiles(p_xp, xp):
    """Sort -log10(p) descending; expected -log10(uniform quantiles)."""
    log10p_obs = -xp.log10(xp.clip(p_xp, 1e-300, 1.0))
    log10p_obs = xp.sort(log10p_obs)[::-1]
    n = log10p_obs.shape[0]
    ranks = xp.arange(1, n + 1, dtype=xp.float64)
    log10p_exp = -xp.log10(ranks / (n + 1.0))
    return log10p_obs, log10p_exp


def _qq_single(
    sumstats: Union[str, Path, pd.DataFrame],
    output: Optional[Union[str, Path]] = None,
    *,
    p_col: str = "P",
    z_col: str = "Z",
    maf_col: str = "MAF",
    maf_bins: Optional[Sequence[float]] = None,
    title: Optional[str] = None,
    xmax: float = 7.2,
    ymax: Optional[float] = None,
    dpi: int = 150,
    use_gpu: bool = True,
    device: int = 0,
):
    """GPU-native QQ plot of -log10(P).

    Parameters
    ----------
    sumstats : path or DataFrame
        Must have ``p_col``. ``z_col`` is used for the chi-sq if present
        (preferred, exact). ``maf_col`` is only required for MAF stratification.
    output : path, optional
        Output png/pdf. If None the figure is not written; it is still returned.
    p_col, z_col, maf_col : str
        Column names.
    maf_bins : sequence of float, optional
        Edges in MAF (e.g. ``[1e-4, 1e-3, 1e-2, 0.05, 0.5]``). When set, the
        plot overlays one QQ curve per stratum.
    title : str
    xmax : float
        x-axis cap for expected -log10(P). Default 7.2 (manuscript style for
        ~1M-variant tests).
    ymax : float
        y-axis cap; default ``max(obs.max() * 1.05, xmax)``.
    dpi : int
    use_gpu : bool
        If False or CuPy unavailable, falls back to numpy.
    device : int
        CUDA device.

    Returns
    -------
    dict with keys
        ``lambda_gc`` : float
        ``mean_chisq`` : float
        ``n`` : int
        ``log10p_obs``, ``log10p_exp`` : np.ndarray
        ``fig`` : matplotlib.figure.Figure
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if HAS_CUPY and use_gpu:
        cp.cuda.Device(device).use()
        xp = cp
    else:
        xp = np

    if isinstance(sumstats, (str, Path)):
        cols = {p_col}
        if maf_bins is not None:
            cols.add(maf_col)
        # always try to read Z if present — efficient chi-sq
        df0 = pd.read_csv(sumstats, sep="\t", nrows=1)
        if z_col in df0.columns:
            cols.add(z_col)
        df = pd.read_csv(sumstats, sep="\t", usecols=list(cols))
    else:
        df = sumstats

    p_np = df[p_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(p_np) & (p_np > 0)
    p_np = p_np[valid]
    p_xp = _to_xp(p_np, xp)

    if z_col in df.columns:
        z_np = df[z_col].to_numpy(dtype=np.float64)[valid]
        chisq_xp = _to_xp(z_np, xp) ** 2
    else:
        # chi-sq from p via inverse CDF — bound to df=1
        # cupy.scipy.stats may not be available; use the approximation
        # chisq ≈ -2 ln P for general use is wrong (df=2). For df=1
        # there is no closed form using numpy without scipy. Use the
        # standard ppf via scipy on the CPU only when Z is missing.
        from scipy.stats import chi2
        chisq_np = chi2.isf(p_np, df=1)
        chisq_xp = _to_xp(chisq_np, xp)

    lambda_gc = _lambda_gc_from_chisq(chisq_xp, xp)
    mean_chisq = float(xp.mean(chisq_xp))

    log10p_obs, log10p_exp = _qq_quantiles(p_xp, xp)

    strata = []
    if maf_bins is not None and maf_col in df.columns:
        maf_np = df[maf_col].to_numpy(dtype=np.float64)[valid]
        maf_xp = _to_xp(maf_np, xp)
        for lo, hi in zip(maf_bins[:-1], maf_bins[1:]):
            mask = (maf_xp >= lo) & (maf_xp < hi)
            n_s = int(mask.sum())
            if n_s < 100:
                continue
            p_s = p_xp[mask]
            lp_o, lp_e = _qq_quantiles(p_s, xp)
            strata.append(dict(
                lo=float(lo), hi=float(hi), n=n_s,
                log10p_obs=cp.asnumpy(lp_o) if xp is cp else lp_o,
                log10p_exp=cp.asnumpy(lp_e) if xp is cp else lp_e,
            ))

    lp_obs_np = cp.asnumpy(log10p_obs) if xp is cp else log10p_obs
    lp_exp_np = cp.asnumpy(log10p_exp) if xp is cp else log10p_exp
    if ymax is None:
        ymax = float(max(lp_obs_np.max() * 1.05, xmax))

    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    if strata:
        cmap = plt.colormaps.get_cmap("viridis")
        for i, s in enumerate(strata):
            ax.scatter(
                s["log10p_exp"], s["log10p_obs"],
                s=2, alpha=0.6,
                color=cmap(i / max(1, len(strata) - 1)),
                label=f"MAF [{s['lo']:.0e}, {s['hi']:.0e})  n={s['n']:,}",
            )
    ax.scatter(lp_exp_np, lp_obs_np, s=2, alpha=0.5, color="#222",
               label=f"all  n={len(lp_obs_np):,}")

    lim = max(xmax, ymax)
    ax.plot([0, lim], [0, lim], color="red", linewidth=0.8, linestyle="--")
    gws = -np.log10(5e-8)
    ax.axhline(gws, color="grey", linestyle=":", linewidth=0.6)

    ax.set_xlim(0, xmax)
    ax.set_ylim(0, ymax)
    ax.set_xlabel(r"Expected $-\log_{10}(P)$")
    ax.set_ylabel(r"Observed $-\log_{10}(P)$")
    ttl = title or "QQ plot"
    ax.set_title(
        f"{ttl}\n"
        f"λ_GC = {lambda_gc:.3f}   "
        f"mean χ² = {mean_chisq:.2f}   "
        f"n = {len(lp_obs_np):,}",
        fontsize=10,
    )
    ax.legend(loc="upper left", fontsize=7, frameon=False)
    ax.set_aspect("equal")
    fig.tight_layout()

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[qq] wrote {out}  λ_GC={lambda_gc:.3f}  "
              f"mean_chisq={mean_chisq:.2f}  n={len(lp_obs_np):,}",
              flush=True)

    return dict(
        lambda_gc=lambda_gc,
        mean_chisq=mean_chisq,
        n=int(len(lp_obs_np)),
        log10p_obs=lp_obs_np,
        log10p_exp=lp_exp_np,
        fig=fig,
    )


# ----------------------------------------------------------------------------
# Batched dispatch (list / dict input + n_workers)
# ----------------------------------------------------------------------------
def _qq_worker(payload):
    """Top-level worker for multiprocessing.Pool(spawn).

    payload = (label, sumstats, output, kw). Re-imports the package in
    the subprocess (each worker is a fresh Python).
    """
    label, sumstats, output, kw = payload
    # Drop the matplotlib figure from the result before returning — Figure
    # objects don't pickle cleanly back through the pool.
    result = _qq_single(sumstats, output, **kw)
    result.pop("fig", None)
    return label, result


def _looks_like_single_sumstats(x) -> bool:
    """Heuristic: a path or DataFrame counts as a single input."""
    return isinstance(x, (str, Path, pd.DataFrame))


def qq(
    sumstats,
    output=None,
    *,
    n_workers: int = 1,
    chunksize: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename_template: str = "qq_{label}.png",
    **kw,
):
    """GPU-native QQ plot, with optional batching across many sumstats.

    Single-input forms (returns one result dict):
      * ``qq(path, output_png)``
      * ``qq(df, output_png)``

    List form (returns ``list[dict]``):
      * ``qq([s1, s2, ...], [o1, o2, ...])``

    Dict form — labels become filenames (returns ``dict[label, dict]``):
      * ``qq({'height': p1, 'bmi': p2}, output_dir='qc/')``
        writes ``qc/qq_INI50.png``, ``qc/qq_BMI.png``.
      * ``filename_template`` defaults to ``qq_{label}.png``.

    Multi-worker fan-out (any batched form):
      * ``n_workers > 1`` spawns a ``Pool(spawn)`` with that many workers.
        Same pattern as ``cg.gwas`` — pairs with CUDA MPS on a single GPU
        for concurrent execution. Single-process default keeps the
        existing behaviour unchanged.

    All other kwargs (``p_col, z_col, maf_col, maf_bins, title, xmax,
    ymax, dpi, use_gpu, device``) are forwarded unchanged to the
    per-pheno worker — except ``title`` is auto-suffixed with the label
    when running on a dict input.
    """
    # ---- dispatch ----
    if _looks_like_single_sumstats(sumstats):
        return _qq_single(sumstats, output, **kw)

    if isinstance(sumstats, Mapping):
        if output_dir is None:
            raise TypeError(
                "qq(dict): output_dir is required (filenames are derived "
                "from dict keys via filename_template)."
            )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = list(sumstats.keys())
        inputs = [sumstats[k] for k in labels]
        outputs = [out_dir / filename_template.format(label=k) for k in labels]
    elif isinstance(sumstats, (list, tuple)):
        if not isinstance(output, (list, tuple)) or len(output) != len(sumstats):
            raise TypeError(
                "qq(list): output must be a matching-length list of paths."
            )
        labels = [str(i) for i in range(len(sumstats))]
        inputs = list(sumstats)
        outputs = list(output)
    else:
        raise TypeError(
            f"qq: unsupported sumstats type {type(sumstats)!r}. "
            f"Use path, DataFrame, list, or dict."
        )

    base_title = kw.pop("title", None)

    def _label_kw(label):
        k = dict(kw)
        if base_title is not None:
            k["title"] = f"{base_title} — {label}"
        elif isinstance(sumstats, Mapping):
            k["title"] = str(label)
        return k

    payloads = [
        (label, ss, out, _label_kw(label))
        for label, ss, out in zip(labels, inputs, outputs)
    ]

    # ---- execute ----
    if n_workers > 1 and len(payloads) > 1:
        # Default chunksize gives each worker one contiguous block so the
        # spawn + GPU + matplotlib warm-up is paid once per worker rather
        # than once per task. (Python's Pool.map default is a much smaller
        # chunksize tuned for load balancing — wrong target here.)
        n_eff = min(n_workers, len(payloads))
        cs = chunksize if chunksize is not None else max(
            1, (len(payloads) + n_eff - 1) // n_eff
        )
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_eff) as pool:
            results = pool.map(_qq_worker, payloads, chunksize=cs)
    else:
        results = [_qq_worker(p) for p in payloads]

    if isinstance(sumstats, Mapping):
        return {label: r for label, r in results}
    return [r for _, r in results]
