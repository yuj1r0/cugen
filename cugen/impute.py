"""cugen.impute - GPU genotype imputation from a phased reference panel.

    impute(target, ref=..., annotation=..., map=...)   -> pd.DataFrame

Ports the imputation stage of Beagle 5 (Browning, Zhou & Browning 2018, AJHG
103(3):338-348) to packed .cugen genotypes.

WHAT THIS DOES AND DOES NOT DO
-------------------------------
Beagle is two algorithms with two papers. This is the second one:

    phasing      Browning et al. (2021) AJHG 108(10):1880-1890   NOT HERE
    imputation   Browning et al. (2018) AJHG 103(3):338-348      this module

That split is Beagle's own, not a simplification imposed on it. The 2018 paper
opens by assuming it: "We assume that the reference and target genotypes are
phased and non-missing. This reduces the computational complexity of imputation
from quadratic to linear in the size of the reference panel." So BOTH inputs
here must be phased .cugen files (encoding hap2bit). Pre-phase the target with
Beagle, SHAPEIT or Eagle first -- the field routinely mixes phasers with
imputers, and the two stages are separately swappable.

Consequence worth stating plainly: an unphased .cugen cannot be imputed by this
module, and it will say so rather than guessing a phase.

WHY THIS IS A GOOD FIT FOR A GPU
---------------------------------
Target haplotypes are completely independent of one another -- the only
sequential axis is the marker axis, and the transition matrix is a diagonal
plus a rank-one term, so each marker costs O(K) rather than O(K^2). That is one
block per target haplotype, K threads, a block reduction per marker.

WHERE THE TIME ACTUALLY GOES
-----------------------------
Sized on the paper's own chr20 fixture before any kernel was written:

    forward-backward       1.1e11
    dose, dense            1.7e12     15x the forward-backward
    dose, sparse       7e9 - 3e10     50-245x cheaper, identical answer

so the dominant phase is turning posteriors into allele probabilities, and it
is dominated in turn by the fact that most reference haplotypes carry the major
allele at any marker. Summing over minor-allele carriers only is what bref3's
allele coding exists to enable, and it is on by default here. Per-phase timings
print under verbose; do not optimise this module without reading them first.

STATE SPACE
-----------
Beagle reduces the state space to imp-states=1600 composite reference
haplotypes assembled from IBS segments. This module currently uses EVERY
reference haplotype as a state, which is exact -- there is no selection
heuristic to be wrong -- and viable on a GPU at 1000-Genomes panel size (4,904
haplotypes against Beagle's 1,600, about 3x). It does NOT reach HRC (26,165
samples) or TOPMed. PBWT-based selection is the next piece of work, and
build_carriers()/the dose kernels are written so it can be dropped in.

WINDOWS
-------
A 40 cM sliding window with 2 cM of overlap. Markers in the overlap are imputed
twice, and the retained value is the one from the window in which the position
is FURTHER from a window boundary -- imputation accuracy degrades towards the
edges. That is equivalent to switching windows at the midpoint of the overlap.

The 2018 paper gives the overlap default as 4 cM and the 5.5 manual as 2.0.
Measured rather than chosen: Beagle 5.5's own chr20 windows convert to
[-0.00, 39.99], [37.99, 77.99] and [75.99, 108.29] cM, so its overlap is
2.00 cM and its window 40.00. cugen's plan_windows produces
[-0.0, 40.0], [38.0, 78.0], [76.0, 116.0] on the same map, which is an
independent check on the windowing as well as on the default.

OUTPUT
------
Per-marker DR2, AF and IMP, matching the INFO fields Beagle writes, plus
optional dosages to a .cugen. Dosages use ENCODING_FLOAT16, which the format
already supports, so imputed output needs no new on-disk work.

References
----------
Browning, Zhou & Browning (2018) AJHG 103(3):338-348   imputation, this module
Browning, Tian, Zhou & Browning (2021) AJHG 108:1880   phasing, not this module
Li & Stephens (2003) Genetics 165(4):2213-2233         the haplotype-copying HMM
"""
import time

import numpy as np
import pandas as pd

from ._genmap import resolve_map
from ._impute_core import (aggregate_markers, allele_sequence_codes,
                           build_carriers, default_err, dose_dense,
                           dose_sparse, forward_backward_blocked,
                           interpolation_weights, transition_tau)
from .io import read_cugen_header
from .write import ENCODING_FLOAT16

__all__ = ["impute", "plan_windows", "dosage_r2"]

_DEFAULT_WINDOW_CM = 40.0
_DEFAULT_OVERLAP_CM = 2.0
_DEFAULT_CLUSTER_CM = 0.005
_DEFAULT_NE = 100_000

# Median per-marker switch probability above which the state sequence is close
# to memoryless and imputation degrades badly. Module-level and therefore
# injectable, so a test can force the warning without building a large panel.
_TAU_WARN = 0.10


def plan_windows(cm, window=_DEFAULT_WINDOW_CM, overlap=_DEFAULT_OVERLAP_CM):
    """Sliding cM windows, and which window owns each marker.

    Returns (bounds, owner) where bounds[w] = (lo_cm, hi_cm) and owner[i] is the
    index of the window whose imputed value is kept for marker i.

    Ownership implements "we discard the imputed value from the window in which
    the position is closer to the window boundary". Because adjacent windows
    overlap by a fixed amount, that rule is exactly a switch at the midpoint of
    the overlap, which is what this computes -- the equivalence is worth
    stating, since a reader checking the paper will look for a distance
    comparison and find arithmetic instead.
    """
    cm = np.asarray(cm, dtype=np.float64)
    if cm.size == 0:
        return [], np.empty(0, dtype=np.int64)
    if window <= overlap:
        raise ValueError(f"window ({window}) must exceed overlap ({overlap})")
    lo0, hi0 = float(cm[0]), float(cm[-1])
    step = window - overlap
    bounds = []
    lo = lo0
    while True:
        hi = lo + window
        bounds.append((lo, hi))
        if hi >= hi0:
            break
        lo += step
    if len(bounds) == 1:
        return bounds, np.zeros(cm.size, dtype=np.int64)

    # switch points at the midpoints of successive overlaps
    owner = np.zeros(cm.size, dtype=np.int64)
    switch = np.array([bounds[w][0] + overlap / 2.0
                       for w in range(1, len(bounds))], dtype=np.float64)
    owner = np.searchsorted(switch, cm, side="right").astype(np.int64)
    return bounds, owner


def dosage_r2(allele_prob):
    """Beagle's DR2: estimated squared correlation of imputed with true dose.

    allele_prob : (T, M) per-haplotype P(allele 1)
    returns     : (M,) in [0, 1]

    Estimated as the fraction of the total dose variance that is between
    haplotypes rather than within them,

        DR2 = Var(a) / (Var(a) + mean(a(1-a)))

    a standard estimator for imputation accuracy without truth. Beagle's exact
    DR2 estimator has not been checked against the binary yet, so treat this as
    a comparable quantity rather than a reproduction of theirs until it has.
    """
    a = np.asarray(allele_prob, dtype=np.float64)
    if a.shape[0] < 2:
        return np.zeros(a.shape[1], dtype=np.float64)
    between = a.var(axis=0)
    within = (a * (1.0 - a)).mean(axis=0)
    tot = between + within
    return np.where(tot > 0, between / np.where(tot > 0, tot, 1.0), 0.0)


def _open_phased(path, what):
    """Open a .cugen and insist it is phased, with an actionable message."""
    from .io import CugenReader
    h = read_cugen_header(path)
    r = CugenReader(path)
    if not r.phased:
        r.close()
        raise ValueError(
            f"{what} {path} is not phased (encoding {h.get('encoding')}). "
            f"Imputation needs haplotypes on both sides: Beagle's own "
            f"imputation stage assumes 'the reference and target genotypes are "
            f"phased and non-missing'. Convert a phased VCF with "
            f"cg.convert.vcf2cugenh(), or pre-phase with Beagle/SHAPEIT/Eagle "
            f"first.")
    return r


def _match_markers(tgt_gidx, ref_gidx):
    """Index of each target marker within the reference, via gidx."""
    order = np.argsort(ref_gidx, kind="stable")
    pos = np.searchsorted(ref_gidx[order], tgt_gidx)
    pos = np.clip(pos, 0, ref_gidx.size - 1)
    idx = order[pos]
    ok = ref_gidx[idx] == tgt_gidx
    if not ok.all():
        missing = int((~ok).sum())
        first = int(tgt_gidx[~ok][0])
        raise ValueError(
            f"{missing:,} target marker(s) are absent from the reference panel, "
            f"first gidx {first}. Beagle requires target markers to be a subset "
            f"of reference markers; drop them or use a matching panel. "
            f"(If the two files were built from different variant universes "
            f"their gidx spaces are not comparable at all.)")
    if np.any(np.diff(idx) <= 0):
        raise ValueError(
            "target markers do not map to strictly increasing reference "
            "positions; both files must be in the same marker order.")
    return idx


def impute(target, *, ref, annotation=None, map=None, out=None,
           ne=_DEFAULT_NE, err=None, window=_DEFAULT_WINDOW_CM,
           overlap=_DEFAULT_OVERLAP_CM, cluster=_DEFAULT_CLUSTER_CM,
           chrom=None, block=None, sparse=True, imp_states=None,
           imp_step=0.1, backend="auto", device=0,
           verbose=True, _timers_out=None):
    """Impute ungenotyped markers into `target` from the phased panel `ref`.

    target, ref : paths to PHASED .cugen files (encoding hap2bit)
    annotation  : DataFrame with `gidx` and `POS` for the REFERENCE markers,
                  optionally CHR/ID; required, because genetic distance is a
                  function of physical position and there is no honest default
    map         : PLINK-format genetic map, a GeneticMap, or None for Beagle's
                  1 cM/Mb assumption
    out         : optional .cugen path for per-sample dosages (float16)

    Returns a DataFrame with one row per reference marker: CHR, POS, ID, AF,
    DR2 and IMP, matching the INFO fields Beagle writes.
    """
    t_all = time.perf_counter()
    # `_timers_out` lets a caller read the per-phase seconds without parsing
    # stdout; benchmarks need the phase split, not just the total.
    timers = {} if _timers_out is None else _timers_out

    if backend not in ("auto", "gpu", "numpy"):
        raise ValueError(f"backend must be auto|gpu|numpy, got {backend!r}")
    if annotation is None:
        raise ValueError(
            "annotation is required: it supplies POS for the reference "
            "markers, and genetic distance -- which drives every transition "
            "probability -- is a function of physical position. There is no "
            "safe default to fall back on.")

    rref = _open_phased(ref, "reference")
    rtgt = _open_phased(target, "target")
    try:
        ann = annotation
        if "gidx" not in ann.columns or "POS" not in ann.columns:
            raise ValueError(
                f"annotation needs 'gidx' and 'POS' columns, has "
                f"{list(ann.columns)}")
        ann = ann.set_index("gidx").reindex(rref.gidx)
        if ann["POS"].isna().any():
            n = int(ann["POS"].isna().sum())
            raise ValueError(
                f"annotation is missing POS for {n:,} of the reference "
                f"panel's {rref.n_variants:,} markers")
        pos = ann["POS"].to_numpy(dtype=np.int64)

        gmap = resolve_map(map, chrom=chrom)
        marker_cm = gmap.cm(pos)

        tgt_idx = _match_markers(rtgt.gidx, rref.gidx)
        K = 2 * rref.n_samples
        T = 2 * rtgt.n_samples
        if err is None:
            err = default_err(K)

        bounds, owner = plan_windows(marker_cm, window, overlap)

        # tau = 1 - exp(-4*Ne*d/|H|) scales INVERSELY with panel size, so
        # Beagle's ne=100000 -- calibrated for panels of tens of thousands of
        # haplotypes -- makes the chain re-randomise at nearly every marker on a
        # small panel. Nothing raises and nothing is NaN; accuracy just
        # collapses, and the output looks like ordinary imputation. Measured on
        # a 400-haplotype panel: median tau 0.60 and dosage r2 0.56 at
        # ne=100000, against tau 0.07 and r2 0.95 at the panel-scaled
        # ne=8000. Beagle's manual gestures at this ("It may be beneficial to
        # specify an appropriate effective population size ... in a small or
        # inbred population") without saying how to tell. This does.
        _, _, _agg = aggregate_markers(marker_cm[tgt_idx], cluster)
        median_tau = float(np.median(transition_tau(_agg / 100.0, ne, K)[1:])) \
            if _agg.size > 1 else 0.0
        if median_tau > _TAU_WARN:
            # Scale ne by panel size rather than by inverting tau. tau is
            # 1-exp(-x), so linearising it is badly wrong exactly where the
            # warning fires (at tau=0.6 the linear estimate is off by 2x), and
            # |H| is the term actually responsible.
            suggested = max(1, int(round(ne * K / 4904.0)))
            print(f"[impute] WARNING: median switch probability tau="
                  f"{median_tau:.3f} is high for a {K:,}-haplotype panel. The "
                  f"HMM will re-randomise its state at most markers and "
                  f"accuracy will be poor -- silently, since nothing here "
                  f"fails. ne={ne:,} is Beagle's default for panels of tens of "
                  f"thousands of haplotypes; tau scales as 4*ne*d/|H|, so try "
                  f"ne~{suggested:,} (= ne * {K:,} / 4904, the 1000 Genomes "
                  f"panel that default suits).")

        # Resolve the backend ONCE, up front, and print it. In the clumping
        # work in this repo a fallback branch hardcoded backend="numpy", so
        # backend="gpu" silently ran a whole chromosome on the CPU reference;
        # it was only found when the run took hours. A path that cannot be
        # observed will eventually be the wrong one.
        from ._impute_gpu import HAS_CUPY
        from ._impute_gpu import impute_haplotypes_gpu
        from ._impute_core import impute_haplotypes as impute_haplotypes_cpu
        if backend == "gpu" and not HAS_CUPY:
            raise RuntimeError(
                "backend='gpu' was requested but CuPy is not importable. "
                "Refusing to fall back to the CPU reference silently -- pass "
                "backend='numpy' if that is what you want.")
        if backend == "auto" and HAS_CUPY:
            # importable is not usable: a CPU node with cupy installed has no device
            from ._device import is_available as _cuda_usable
            auto_gpu = _cuda_usable()
        else:
            auto_gpu = False
        use_gpu = (backend == "gpu") or (backend == "auto" and auto_gpu)
        engine = impute_haplotypes_gpu if use_gpu else impute_haplotypes_cpu
        path = "gpu (fused kernels)" if use_gpu else "numpy reference (CPU)"
        if verbose:
            print(f"[impute] reference {rref.n_samples:,} samples "
                  f"({K:,} haplotypes) x {rref.n_variants:,} markers")
            print(f"[impute] target    {rtgt.n_samples:,} samples "
                  f"({T:,} haplotypes) x {rtgt.n_variants:,} markers")
            print(f"[impute] map {gmap} | ne={ne:,} err={err:.3e} "
                  f"cluster={cluster} cM")
            print(f"[impute] {len(bounds)} window(s) of {window} cM, "
                  f"{overlap} cM overlap | dose={'sparse' if sparse else 'dense'}")
            print(f"[impute] path: {path}")

        t0 = time.perf_counter()
        # The target is small (an array marker set), so it is read whole. The
        # reference is not: at chr20 scale the full haplotype matrix is 8.5 GiB,
        # and slicing a window out of it costs another copy on top. Windows are
        # contiguous in position, so each one is a ranged read instead, and peak
        # memory becomes a function of WINDOW size rather than chromosome size.
        tgt_bits_all = rtgt.read_haplotypes()
        timers["read"] = time.perf_counter() - t0

        # Per-sample dosages in float32, plus per-marker summaries. NOT a
        # full-chromosome per-HAPLOTYPE float64 array: that is O(T * M * 8) and
        # was 27.7 GiB at 2,000 target haplotypes on chr20, with `dose` adding
        # another 13.9 GiB on top. Measured RSS hit 60 GiB and the run went from
        # GPU-bound to thrashing. Allele probabilities are now window-scoped and
        # the per-marker statistics are accumulated as each window completes.
        dose = np.zeros((rtgt.n_samples, rref.n_variants), dtype=np.float32)
        af = np.zeros(rref.n_variants, dtype=np.float64)
        dr2 = np.zeros(rref.n_variants, dtype=np.float64)
        tgt_owner = owner[tgt_idx]

        for w, (lo, hi) in enumerate(bounds):
            mk = np.flatnonzero((marker_cm >= lo) & (marker_cm <= hi))
            tk = np.flatnonzero((marker_cm[tgt_idx] >= lo) &
                                (marker_cm[tgt_idx] <= hi))
            if mk.size == 0 or tk.size < 2:
                continue
            keep = np.flatnonzero(owner[mk] == w)
            if keep.size == 0:
                continue
            # mk is contiguous because markers are position-sorted and a
            # window is a cM interval; assert it rather than assume, since a
            # non-contiguous mk would make the ranged read silently return the
            # wrong markers instead of failing.
            m0, m1 = int(mk[0]), int(mk[-1]) + 1
            assert mk.size == m1 - m0, "window markers are not contiguous"
            t_read = time.perf_counter()
            if use_gpu:
                # Packed bytes only. The device builds the carrier lists from
                # them and unpacks just the genotyped columns, so the window is
                # never expanded to a (K, M) byte matrix on the host -- 2.4 GiB
                # per window at chr20 scale, read and scanned for nothing.
                sub_ref = None
                sub_packed = np.frombuffer(rref.read_packed_bytes(m0, m1),
                                           dtype=np.uint8)
            else:
                sub_ref = rref.read_haplotypes(m0, m1)
                sub_packed = None
            timers["read"] = timers.get("read", 0.0) + (time.perf_counter() - t_read)
            sub_tgt = tgt_bits_all[:, tk]
            # tgt_idx points into the full reference; re-index into this window
            loc = tgt_idx[tk] - m0
            kw = dict(ne=ne, err=err, cluster=cluster, timers=timers,
                      imp_states=imp_states, imp_step=imp_step)
            if use_gpu:
                kw.update(ref_packed=sub_packed, n_hap=K,
                          bytes_per_variant=rref.bytes_per_variant,
                          reduce=True)
            else:
                kw.update(block=block, sparse=sparse)
            if use_gpu:
                # The device returns per-SAMPLE dosages and the two per-marker
                # summaries directly. The per-haplotype array is 7.3 GiB at
                # 2,000 target haplotypes over chr20's largest window and never
                # crosses to the host.
                dose_w, af_w, dr2_w = engine(sub_ref, sub_tgt, loc,
                                             marker_cm[mk], **kw)
                dose[:, mk[keep]] = dose_w[:, keep]
                af[mk[keep]] = af_w[keep]
                dr2[mk[keep]] = dr2_w[keep]
                del dose_w, af_w, dr2_w
            else:
                pw = engine(sub_ref, sub_tgt, loc, marker_cm[mk], **kw)
                kept = pw[:, keep]
                dose[:, mk[keep]] = (kept[0::2, :] + kept[1::2, :]
                                     ).astype(np.float32)
                af[mk[keep]] = kept.mean(axis=0)
                dr2[mk[keep]] = dosage_r2(kept)
                del kept, pw
            if verbose:
                print(f"[impute]   window {w+1}/{len(bounds)} "
                      f"[{lo:.1f}, {hi:.1f}] cM: {mk.size:,} markers, "
                      f"{tk.size:,} genotyped, {keep.size:,} kept")

        t0 = time.perf_counter()
        is_typed = np.zeros(rref.n_variants, dtype=bool)
        is_typed[tgt_idx] = True
        timers["summary"] = time.perf_counter() - t0

        if out is not None:
            t0 = time.perf_counter()
            from .write import CugenWriter
            with CugenWriter(out, rtgt.n_samples, rref.n_variants,
                             encoding=ENCODING_FLOAT16) as w_:
                # In blocks, not per variant: the per-variant loop measured 32s
                # for one chromosome, more than the entire GPU computation that
                # produced the values.
                step = 200_000
                for j0 in range(0, rref.n_variants, step):
                    j1 = min(j0 + step, rref.n_variants)
                    w_.add_variants_bulk(rref.gidx[j0:j1], dose[:, j0:j1])
            timers["write"] = time.perf_counter() - t0

        frame = pd.DataFrame({
            "gidx": rref.gidx,
            "CHR": ann["CHR"].to_numpy() if "CHR" in ann.columns else "NA",
            "POS": pos,
            "ID": ann["ID"].to_numpy() if "ID" in ann.columns
                  else np.arange(rref.n_variants),
            "CM": marker_cm,
            "AF": af,
            "DR2": dr2,
            "IMP": ~is_typed,
        })

        if verbose:
            total = time.perf_counter() - t_all
            print(f"[impute] phases: " + "  ".join(
                f"{k} {v:.2f}s" for k, v in sorted(((k, v) for k, v in timers.items()
                         if not k.startswith("_")),
                        key=lambda kv: -kv[1])))
            print(f"[impute] {int((~is_typed).sum()):,} markers imputed, "
                  f"{int(is_typed.sum()):,} genotyped, {total:.2f}s total")
        return frame
    finally:
        rref.close()
        rtgt.close()
