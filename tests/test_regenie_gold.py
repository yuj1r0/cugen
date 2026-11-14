"""CuGenie against a REGENIE v2.2.4 step-1 gold fixture (CPU).

The fixture (tests/data/regenie_fixture.* and regenie_gold_1.loco) is a synthetic cohort
of 300 samples and 150 variants with missing calls; tests/data/README.md has the commands
that regenerate it. REGENIE writes the LOCO offsets as text with ~6 significant digits.
"""
import os

import numpy as np
import pandas as pd
import pytest

import cugen as cg

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
PRE = os.path.join(DATA, "regenie_fixture")


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    monkeypatch.setenv("CUGEN_DEVICE", "cpu")


def read_loco(path):
    """REGENIE .loco: header 'FID_IID <fid>_<iid> ...', then one row per chromosome."""
    with open(path) as fh:
        hdr = fh.readline().split()
        rows = [ln.split() for ln in fh if ln.strip()]
    ids = [h.split("_")[0] for h in hdr[1:]]
    return ids, [int(r[0]) for r in rows], np.array([[float(v) for v in r[1:]] for r in rows])


def test_cugenie_masked_loco_matches_regenie():
    geno = cg.BedGenotypes(PRE)
    ph = pd.read_csv(PRE + ".pheno", sep=r"\s+")
    cv = pd.read_csv(PRE + ".covar", sep=r"\s+")
    assert ph.IID.astype(str).tolist() == geno.sample_ids.tolist()
    loco = cg.fit_cugenie(geno, ph["y"].to_numpy(), cv[["c1", "c2"]].to_numpy(), block_size=25,
                          verbose=False)
    ids, chrs, X = read_loco(os.path.join(DATA, "regenie_gold_1.loco"))
    pos = {s: i for i, s in enumerate(ids)}                  # REGENIE writes its own sample order
    ix = np.array([pos[s] for s in geno.sample_ids])
    assert loco.chromosomes.tolist() == [1, 2, 3]
    for c in (1, 2, 3):
        ours = loco.offsets[:, loco.column(c), 0]
        gold = X[chrs.index(c), ix]
        assert np.corrcoef(ours, gold)[0, 1] > 0.999999
        np.testing.assert_allclose(ours, gold, rtol=0, atol=1e-5)
    # level-1 cross-validation, as printed in REGENIE's log (h2 = 0.01, 0.25, 0.5, 0.75, 0.99)
    l1 = loco.meta["level1"]["y"]
    assert l1["tau_index"] == 1
    np.testing.assert_allclose(l1["rsq"], [0.142428, 0.136836, 0.130887, 0.121389, 0.0991789], rtol=5e-6)
    np.testing.assert_allclose(l1["mse"], [0.92057, 0.856889, 0.867956, 0.883516, 0.923044], rtol=5e-6)
