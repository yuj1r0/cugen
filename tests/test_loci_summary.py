"""loci: one lambda_GC, one distance-clump core; define_loci output unchanged."""
import numpy as np
import pandas as pd
import pytest

import cugen as cg
from cugen import loci


def test_constants_are_exact():
    from scipy.stats import chi2, norm
    assert loci.CHI2_MEDIAN == pytest.approx(chi2.ppf(0.5, 1), rel=1e-15)
    assert loci.ZGWS == pytest.approx(norm.isf(2.5e-8), rel=1e-15)


def test_lambda_gc_z_and_p_agree():
    from scipy.stats import chi2, norm
    rng = np.random.default_rng(0)
    for m in (10001, 10000):                                 # odd and even counts
        z = rng.normal(size=m) * 1.1
        p = 2 * norm.sf(np.abs(z))
        lz = cg.lambda_gc(z=z)
        lp = cg.lambda_gc(p=p)
        ref = np.median(chi2.isf(p, 1)) / chi2.ppf(0.5, 1)
        assert lp == pytest.approx(ref, rel=1e-12)
        assert lz == pytest.approx(lp, rel=1e-6)
    assert abs(cg.lambda_gc(rng.normal(size=200000)) - 1.0) < 0.02
    assert np.isnan(cg.lambda_gc(z=[np.nan]))
    with pytest.raises(ValueError):
        cg.lambda_gc()


def test_summarize():
    chrom = np.r_[np.ones(5), np.full(3, 2)].astype(int)
    pos = np.array([100, 200, 3_000_000, 3_500_000, 9_000_000, 10, 20, 30])
    zz = np.array([6, 7, 8, 1, 6, 9, 6, 6.0])
    info, cl = cg.summarize(z=zz, chrom=chrom, pos=pos)
    assert info["n_gws"] == 7 and info["n_loci"] == 4 and info["n_chrom_with_gws"] == 2
    assert cl.LOG10P.is_monotonic_decreasing
    assert sorted(cl.row.tolist()) == [1, 2, 4, 5]          # leads, as rows of the input
    np.testing.assert_array_equal(pos[cl.row], cl.POS)
    df = pd.DataFrame({"CHR": chrom, "POS": pos, "Z": zz})
    info2, cl2 = cg.summarize(df)
    assert info2 == info
    pd.testing.assert_frame_equal(cl, cl2)


def _old_step2(sig, clump_window):
    """The previous define_loci step 2 (row-by-row), kept as the reference."""
    lead_snps, used = [], set()
    for _, row in sig.iterrows():
        key = (row["CHR"], row["POS"])
        if key in used:
            continue
        lead_snps.append(row)
        mask = ((sig["CHR"] == row["CHR"]) & (sig["POS"] >= row["POS"] - clump_window)
                & (sig["POS"] <= row["POS"] + clump_window))
        for _, r in sig[mask].iterrows():
            used.add((r["CHR"], r["POS"]))
    return lead_snps


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_clump_core_reproduces_define_loci_step2(seed):
    rng = np.random.default_rng(seed)
    m = 400
    pos = rng.integers(1, 20_000_000, m)
    pos[::17] = pos[1::17][: len(pos[::17])]                 # duplicate positions
    df = pd.DataFrame({"CHR": rng.integers(1, 4, m), "POS": pos,
                       "P": np.round(10 ** -rng.uniform(5, 30, m), 25),   # ties
                       "gidx": np.arange(m), "ID": [f"v{i}" for i in range(m)]})
    sig = df[df.P < 5e-8].sort_values("P").reset_index(drop=True)
    old = _old_step2(sig, 1_000_000)
    lead_idx, _ = loci._greedy_clump(sig.CHR.to_numpy(), sig.POS.to_numpy(), np.arange(len(sig)), 1_000_000)
    assert [int(r["gidx"]) for r in old] == sig.gidx.to_numpy()[lead_idx].tolist()


def _old_core(chrom, pos, order, window):
    """The previous step-2 loop on arrays, in the core's calling convention."""
    used, leads = set(), []
    for i in order:
        key = (chrom[i], pos[i])
        if key in used:
            continue
        leads.append(i)
        m = (chrom == chrom[i]) & (pos >= pos[i] - window) & (pos <= pos[i] + window)
        for j in np.flatnonzero(m):
            used.add((chrom[j], pos[j]))
    return np.asarray(leads, dtype=np.int64), None


@pytest.mark.parametrize("merge", [True, False])
def test_define_loci_end_to_end_unchanged(monkeypatch, merge):
    rng = np.random.default_rng(5)
    m = 3000
    df = pd.DataFrame({"CHR": rng.integers(1, 23, m), "POS": rng.integers(1, 50_000_000, m),
                       "P": 10 ** -rng.uniform(0, 20, m), "gidx": np.arange(m),
                       "ID": [f"rs{i}" for i in range(m)]})
    df.loc[df.index[::50], "POS"] = df.POS.iloc[1::50].to_numpy()[: len(df.index[::50])]
    new = cg.define_loci(df, merge_overlapping=merge)
    monkeypatch.setattr(loci, "_greedy_clump", _old_core)
    old = cg.define_loci(df, merge_overlapping=merge)
    assert len(new) > 5
    pd.testing.assert_frame_equal(new, old)
