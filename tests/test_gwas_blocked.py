"""Blocked step 4 (cugen.background.gwas_blocked) against a dense per-variant reference (CPU)."""
import gzip
import os

import numpy as np
import pandas as pd
import pytest

import cugen as cg
from conftest import a2_counts
from cugen import background as bg


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUGEN_DEVICE", "cpu")


@pytest.fixture(scope="module")
def loco(cohort):
    return cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=30, verbose=False,
                          device="cpu")


def ref_step4(A, r, mu=None, sxx=None):
    n = A.shape[1]
    rc = r - r.mean()
    syy = rc @ rc
    out = []
    for v in range(A.shape[0]):
        g = A[v].astype(float)
        val = A[v] != 3
        m_ = g[val].mean() if mu is None else mu[v]
        s_ = ((g[val] - g[val].mean()) ** 2).sum() if sxx is None else sxx[v]
        num = ((g[val] - m_) * rc[val]).sum()
        b = num / s_
        se = np.sqrt((syy - b * b * s_) / ((n - 2) * s_))
        out.append(b / se)
    return np.array(out)


def test_equals_reference_bed_and_cugen(cohort, loco):
    G, chrom = cohort["G"], cohort["chrom"]
    ss = bg.gwas_blocked(cohort["bed"], loco, block_size=37, maf_min=None, verbose=False)
    for c in (1, 2, 3):
        o = chrom == c
        np.testing.assert_allclose(ss.Z.to_numpy()[o], ref_step4(G[o], loco.residual(c)), atol=2e-5, rtol=1e-6)
    sc = bg.gwas_blocked(cohort["cg"], loco, maf_min=None, verbose=False, check_samples=False)
    mu, sxx, _ = cohort["cg"].stored_stats(np.arange(len(chrom)))
    o = chrom == 2
    zr = ref_step4(a2_counts(G)[o], loco.residual(2), mu[o].astype(float), sxx[o].astype(float))
    np.testing.assert_allclose(sc.Z.to_numpy()[o], zr, atol=2e-5, rtol=1e-6)
    np.testing.assert_allclose(sc.Z.to_numpy(), -ss.Z.to_numpy(), atol=1e-4)    # A1 vs A2 counted


def test_production_columns_and_counted_allele(cohort, loco):
    ss = bg.gwas_blocked(cohort["bed"], loco, verbose=False)
    assert list(ss.columns) == ["CHR", "POS", "ID", "REF", "ALT", "BETA", "SE", "Z", "P", "MAF", "gidx"]
    assert ss.attrs["counted_allele"] == "bim A1 (column 5)"
    assert (ss.ALT == "A").all() and (ss.REF == "G").all()             # ALT = counted allele (bim A1)
    assert ss.CHR.dtype == np.int64 and ss.POS.dtype == np.int64 and ss.gidx.dtype == np.int64
    assert ss.BETA.dtype == np.float32 and ss.P.dtype == np.float64


def test_maf_min_default_matches_gwas(cohort, loco):
    import inspect
    assert inspect.signature(bg.gwas_blocked).parameters["maf_min"].default == \
        inspect.signature(cg.gwas).parameters["maf_min"].default
    allv = bg.gwas_blocked(cohort["bed"], loco, maf_min=None, verbose=False)
    some = bg.gwas_blocked(cohort["bed"], loco, maf_min=0.2, verbose=False)
    assert len(some) == int((allv.MAF >= 0.2).sum()) < len(allv)


def test_multi_background_and_roundtrip(cohort, loco, tmp_path):
    K = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], verbose=False)
    p = loco.save(str(tmp_path / "loco"))[0]
    back = bg.LocoPredictions.load(p)
    np.testing.assert_allclose(back.predictions, loco.predictions.astype(np.float32), rtol=0, atol=0)
    multi = bg.gwas_blocked(cohort["bed"], {"L": loco, "K": K}, verbose=False)
    one = bg.gwas_blocked(cohort["bed"], K, verbose=False)
    assert set(multi) == {"L", "K"}
    np.testing.assert_allclose(multi["K"].Z, one.Z, atol=1e-6)
    out = bg.gwas_blocked(cohort["bed"], p, verbose=False, output=str(tmp_path / "s.feather"))
    assert os.path.exists(tmp_path / "s.feather") and len(out) == len(one)


def test_output_layout_matches_production(cohort, loco, tmp_path):
    d = tmp_path / "gwas"
    ss = bg.gwas_blocked(cohort["bed"], loco, verbose=False, output=str(d))
    assert sorted(os.listdir(d)) == ["all_sumstats.tsv.gz", "chr1_sumstats.tsv", "chr2_sumstats.tsv",
                                     "chr3_sumstats.tsv"]
    with gzip.open(d / "all_sumstats.tsv.gz", "rt") as fh:
        allss = pd.read_csv(fh, sep="\t")
    assert len(allss) == len(ss) and list(allss.columns) == list(ss.columns)
    raw = pd.read_csv(d / "chr1_sumstats.tsv", sep="\t", dtype=str)
    assert raw.POS.str.fullmatch(r"\d+").all()                         # exact integers
    f = tmp_path / "one.tsv.gz"
    bg.gwas_blocked(cohort["bed"], loco, verbose=False, output=str(f))
    with gzip.open(f, "rt") as fh:
        assert len(pd.read_csv(fh, sep="\t")) == len(ss)
    multi = tmp_path / "multi"
    bg.gwas_blocked(cohort["bed"], {"a": loco, "b": loco}, verbose=False, output=str(multi))
    assert sorted(os.listdir(multi)) == ["a", "b"]


def test_rejects_chromosomes_missing_from_background(cohort, loco):
    narrow = bg.LocoPredictions(loco.predictions[:, :2, 0], loco.y_original[:, 0], loco.fid_order)
    with pytest.raises(ValueError, match="chromosomes \\[1, 2\\]"):
        bg.gwas_blocked(cohort["bed"], narrow, verbose=False)
    ok = bg.gwas_blocked(cohort["bed"], narrow, chromosomes=[1, 2], verbose=False)
    assert set(ok.CHR.unique()) == {1, 2}
    # a background on a non-contiguous chromosome set is indexed through its chromosomes field
    sub = cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=30, chromosomes=[1, 3],
                         verbose=False)
    s13 = bg.gwas_blocked(cohort["bed"], sub, chromosomes=[1, 3], maf_min=None, verbose=False)
    o = cohort["chrom"] == 3
    np.testing.assert_allclose(s13.Z.to_numpy()[s13.CHR.to_numpy() == 3],
                               ref_step4(cohort["G"][o], sub.residual(3)), atol=2e-5, rtol=1e-6)


def test_sample_order_check(cohort, loco):
    shuffled = bg.LocoPredictions(loco.predictions[:, :, 0], loco.y_original[:, 0], loco.fid_order[::-1])
    with pytest.raises(ValueError, match="fid_order"):
        bg.gwas_blocked(cohort["bed"], shuffled, verbose=False)
