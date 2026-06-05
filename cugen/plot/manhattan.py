"""cugen.plot.manhattan — GPU-native Manhattan plot.

CuPy: -log10(p), per-chr cumulative offsets, decimation; matplotlib renders.

Decimation strategy: keep ALL variants with -log10(P) >= ``keep_above``
(default 3.0, i.e. P <= 1e-3); from the rest, keep a random ``keep_frac``
sample (default 1%). On 6.8M-variant imputed runs this drops scatter density
from ~tens-of-millions to ~hundreds-of-thousands of points with no visible
loss in the manuscript-style plot.

qqman-style loglog y-axis above ``loglog_break`` (default 20) is supported
via ``loglog=True`` — same form as `make_figures_short_letter.py`.

Batched inputs work in the same ``manhattan()`` call:

  * single  — ``manhattan(sumstats, output)``
  * list    — ``manhattan([s1, s2], [o1, o2])``
  * dict    — ``manhattan({'height': p1, 'bmi': p2}, output_dir='qc/')``
              → writes ``qc/manhattan_{label}.png`` for each.

``n_workers > 1`` spawns a ``Pool(spawn)`` for the batched forms (matches
``cg.gwas`` MPS-shared pattern).
"""

import multiprocessing as mp
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


_CHR_COLORS = ["#1f4d80", "#80c1ff"]


def _to_xp(x, xp):
    if xp is cp:
        return cp.asarray(x)
    return np.asarray(x)


def _loglog_transform(y, breakpt):
    """qqman piecewise loglog: y up to breakpt linear, above multiplicative.

    For y > breakpt: y' = breakpt + 10*(log10(y) - log10(breakpt)).
    """
    y = np.asarray(y, dtype=np.float64)
    out = y.copy()
    mask = y > breakpt
    out[mask] = breakpt + 10.0 * (np.log10(y[mask]) - np.log10(breakpt))
    return out


def _manhattan_single(
    sumstats: Union[str, Path, pd.DataFrame],
    output: Optional[Union[str, Path]] = None,
    *,
    chr_col: str = "CHR",
    pos_col: str = "POS",
    p_col: str = "P",
    chromosomes: Sequence = tuple(range(1, 23)),
    keep_above: float = 3.0,
    keep_frac: float = 0.01,
    p_threshold: float = 5e-8,
    title: Optional[str] = None,
    loglog: bool = False,
    loglog_break: float = 20.0,
    figsize: tuple = (12, 4),
    dpi: int = 150,
    use_gpu: bool = True,
    device: int = 0,
    seed: int = 42,
):
    """GPU-native Manhattan plot.

    Parameters
    ----------
    sumstats : path or DataFrame
        Must have ``CHR, POS, P`` columns (case-sensitive).
    output : path, optional
        png/pdf path. If None, only the figure is returned.
    chromosomes : sequence of int
        Chromosomes to plot (default 1..22).
    keep_above : float
        Always keep variants with -log10(P) >= keep_above (default 3.0).
    keep_frac : float
        Fraction of below-threshold variants to randomly subsample
        (default 0.01, ie keep 1%). Set to 1.0 to disable decimation.
    p_threshold : float
        Significance line (default 5e-8).
    loglog : bool
        Use qqman-style piecewise loglog y-axis above ``loglog_break``.
    loglog_break : float
        Breakpoint for the loglog transform (default -log10(P)=20).
    figsize, dpi, use_gpu, device, seed : standard.

    Returns
    -------
    dict with keys
        ``n_total``, ``n_plotted``, ``n_significant``, ``lambda_gc`` (Bayesian
        approx skipped here — see plot.qq for exact), ``fig``.
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
        df = pd.read_csv(
            sumstats, sep="\t",
            usecols=[chr_col, pos_col, p_col],
        )
    else:
        df = sumstats

    chrs_np = df[chr_col].to_numpy()
    pos_np = df[pos_col].to_numpy(dtype=np.int64)
    p_np = df[p_col].to_numpy(dtype=np.float64)
    valid = np.isfinite(p_np) & (p_np > 0) & np.isin(chrs_np, list(chromosomes))
    chrs_np = chrs_np[valid].astype(np.int32)
    pos_np = pos_np[valid]
    p_np = p_np[valid]

    n_total = int(len(p_np))
    p_thresh_logp = -np.log10(p_threshold)

    chr_xp = _to_xp(chrs_np, xp)
    pos_xp = _to_xp(pos_np, xp)
    p_xp = _to_xp(p_np, xp)
    log10p_xp = -xp.log10(xp.clip(p_xp, 1e-300, 1.0))

    n_sig = int((log10p_xp >= -xp.log10(xp.asarray(p_threshold))).sum())

    # Decimation mask: keep all above threshold + random keep_frac below.
    above = log10p_xp >= keep_above
    if keep_frac < 1.0:
        rng_state = cp.random.get_random_state() if xp is cp else None
        if xp is cp:
            cp.random.seed(seed)
            r = cp.random.random(log10p_xp.shape[0])
        else:
            rng = np.random.default_rng(seed)
            r = rng.random(log10p_xp.shape[0])
        below_sample = (~above) & (r < keep_frac)
        keep_mask = above | below_sample
    else:
        keep_mask = xp.ones_like(log10p_xp, dtype=bool)

    chr_xp = chr_xp[keep_mask]
    pos_xp = pos_xp[keep_mask]
    log10p_xp = log10p_xp[keep_mask]

    # Per-chr offsets (cumulative max position per chr)
    chr_max = {}
    for c in chromosomes:
        cm = (chr_xp == int(c))
        if not bool(cm.any()):
            chr_max[int(c)] = 0
            continue
        chr_max[int(c)] = int(pos_xp[cm].max())
    offsets = {}
    running = 0
    pad = 0  # set to a small positive number for visible chr gaps
    for c in chromosomes:
        offsets[int(c)] = running
        running += chr_max[int(c)] + pad
    total_span = running

    # Genome-coordinate vector
    offset_xp = _to_xp(np.array([offsets[int(c)] for c in chromosomes],
                                 dtype=np.int64), xp)
    chr_list = list(chromosomes)
    # map chr_xp -> offset index
    # Simpler: compute per-chr genome coord on the GPU
    gx = xp.zeros_like(pos_xp, dtype=xp.float64)
    for i, c in enumerate(chr_list):
        m = chr_xp == int(c)
        if bool(m.any()):
            gx[m] = pos_xp[m].astype(xp.float64) + float(offsets[int(c)])
    gx_np = cp.asnumpy(gx) if xp is cp else gx
    chr_np = cp.asnumpy(chr_xp) if xp is cp else chr_xp
    lp_np = cp.asnumpy(log10p_xp) if xp is cp else log10p_xp

    n_plotted = int(len(lp_np))
    if loglog:
        lp_plot = _loglog_transform(lp_np, loglog_break)
        gws_y = _loglog_transform(np.asarray([p_thresh_logp]), loglog_break)[0]
    else:
        lp_plot = lp_np
        gws_y = p_thresh_logp

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    xticks = []
    xtick_labels = []
    for i, c in enumerate(chr_list):
        m = chr_np == int(c)
        if not m.any():
            continue
        color = _CHR_COLORS[i % len(_CHR_COLORS)]
        ax.scatter(gx_np[m], lp_plot[m], s=1.5, c=color, alpha=0.7,
                   edgecolors="none", rasterized=True)
        c_mid = offsets[int(c)] + chr_max[int(c)] / 2
        xticks.append(c_mid)
        xtick_labels.append(str(c))
    ax.axhline(gws_y, color="red", linewidth=0.6, linestyle=":")

    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels, fontsize=8)
    ax.set_xlim(0, total_span)
    if loglog:
        # y-axis label ticks: 0, 5, 10, 15, 20, then loglog 50, 100, 500, 1000
        ymax = lp_plot.max() * 1.05 if len(lp_plot) else gws_y * 2
        ax.set_ylim(0, ymax)
        # Add explanatory tick at break
        ax.axhline(loglog_break, color="grey", linewidth=0.4, linestyle="--",
                   alpha=0.5)
    else:
        ax.set_ylim(0, max(lp_plot.max() * 1.05 if len(lp_plot) else 10, 12))

    ax.set_xlabel("Chromosome")
    ax.set_ylabel(r"$-\log_{10}(P)$"
                  + (f"  (loglog above {loglog_break:.0f})" if loglog else ""))
    ttl = title or "Manhattan plot"
    ax.set_title(
        f"{ttl}    n_total={n_total:,}    n_plotted={n_plotted:,}    "
        f"n_signif (P<{p_threshold:.0e})={n_sig:,}",
        fontsize=10,
    )
    fig.tight_layout()

    if output is not None:
        out = Path(output)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"[manhattan] wrote {out}  n_total={n_total:,}  "
              f"n_plotted={n_plotted:,}  n_signif={n_sig:,}",
              flush=True)

    return dict(
        n_total=n_total,
        n_plotted=n_plotted,
        n_significant=n_sig,
        fig=fig,
    )


# ----------------------------------------------------------------------------
# Batched dispatch (list / dict input + n_workers)
# ----------------------------------------------------------------------------
def _manhattan_worker(payload):
    """Top-level worker for multiprocessing.Pool(spawn).

    payload = (label, sumstats, output, kw).
    """
    label, sumstats, output, kw = payload
    result = _manhattan_single(sumstats, output, **kw)
    result.pop("fig", None)
    return label, result


def _looks_like_single_sumstats(x) -> bool:
    return isinstance(x, (str, Path, pd.DataFrame))


def manhattan(
    sumstats,
    output=None,
    *,
    n_workers: int = 1,
    chunksize: Optional[int] = None,
    output_dir: Optional[Union[str, Path]] = None,
    filename_template: str = "manhattan_{label}.png",
    **kw,
):
    """GPU-native Manhattan plot, with optional batching.

    Single-input forms (returns one result dict):
      * ``manhattan(path, output_png)``
      * ``manhattan(df, output_png)``

    List form (returns ``list[dict]``):
      * ``manhattan([s1, s2, ...], [o1, o2, ...])``

    Dict form — labels become filenames (returns ``dict[label, dict]``):
      * ``manhattan({'height': p1, 'bmi': p2}, output_dir='qc/')``
        writes ``qc/manhattan_INI50.png`` etc.

    Multi-worker fan-out (any batched form):
      * ``n_workers > 1`` spawns a ``Pool(spawn)`` — pairs with CUDA MPS
        for concurrent execution on a single GPU.
    """
    if _looks_like_single_sumstats(sumstats):
        return _manhattan_single(sumstats, output, **kw)

    if isinstance(sumstats, Mapping):
        if output_dir is None:
            raise TypeError(
                "manhattan(dict): output_dir is required (filenames are "
                "derived from dict keys via filename_template)."
            )
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        labels = list(sumstats.keys())
        inputs = [sumstats[k] for k in labels]
        outputs = [out_dir / filename_template.format(label=k) for k in labels]
    elif isinstance(sumstats, (list, tuple)):
        if not isinstance(output, (list, tuple)) or len(output) != len(sumstats):
            raise TypeError(
                "manhattan(list): output must be a matching-length list of "
                "paths."
            )
        labels = [str(i) for i in range(len(sumstats))]
        inputs = list(sumstats)
        outputs = list(output)
    else:
        raise TypeError(
            f"manhattan: unsupported sumstats type {type(sumstats)!r}. "
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

    if n_workers > 1 and len(payloads) > 1:
        n_eff = min(n_workers, len(payloads))
        cs = chunksize if chunksize is not None else max(
            1, (len(payloads) + n_eff - 1) // n_eff
        )
        ctx = mp.get_context("spawn")
        with ctx.Pool(n_eff) as pool:
            results = pool.map(_manhattan_worker, payloads, chunksize=cs)
    else:
        results = [_manhattan_worker(p) for p in payloads]

    if isinstance(sumstats, Mapping):
        return {label: r for label, r in results}
    return [r for _, r in results]
