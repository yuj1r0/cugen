"""Phased (haplotype) LD statistics.

The oracle here is a direct numpy correlation of the 0/1 haplotype indicator
columns -- deliberately NOT the production formula, so agreement is evidence
rather than tautology. For phased biallelic data the identity being checked is

    r_phased = D / sqrt(pA qA pB qB),  D = p_AB - pA pB

over H = 2*n_samples haplotypes, which for 0/1 indicators is exactly
Pearson's r on those columns.
"""
import numpy as np
import pytest

from cugen import ld as L
from cugen.write import write_cugen, write_cugen_phased
from conftest import requires_cudf, requires_gpu, simulate_haplotypes

CPU = dict(backend="numpy", verbose=False)


def simulate_phased(n_samples, n_variants, seed=0):
    """(2*n_samples, n_variants) 0/1 alleles with a spread of MAF and LD."""
    rng = np.random.default_rng(seed)
    H = 2 * n_samples
    latent = rng.random(H)
    out = np.zeros((H, n_variants), dtype=np.uint8)
    for v in range(n_variants):
        mix = rng.uniform(0.0, 0.95)
        freq = rng.uniform(0.05, 0.95)
        noise = rng.random(H)
        score = mix * latent + (1 - mix) * noise
        out[:, v] = (score < freq).astype(np.uint8)
    return out


def oracle_phased_r(hap, i, j):
    """Pearson r of two 0/1 haplotype indicator columns."""
    x = hap[:, i].astype(np.float64)
    y = hap[:, j].astype(np.float64)
    if x.std() == 0 or y.std() == 0:
        return np.nan
    return float(np.corrcoef(x, y)[0, 1])


def oracle_phased_d(hap, i, j):
    """D = p_AB - pA*pB over haplotypes."""
    x = hap[:, i].astype(np.float64)
    y = hap[:, j].astype(np.float64)
    return float((x * y).mean() - x.mean() * y.mean())


def test_r_phased_matches_haplotype_correlation(tmp_path):
    """r_phased on a hap2bit file is Pearson r of the haplotype indicators."""
    hap = simulate_phased(40, 12, seed=3)
    p = tmp_path / "phased.cugen"
    write_cugen_phased(str(p), hap)

    df = L.ld_matrix(str(p), stats=("r_phased", "r2_phased"), min_r2=0.0,
                     min_obs=1, **CPU)

    # a monomorphic variant has zero variance, so its pairs are NaN and get
    # filtered; derive the expectation instead of assuming full polymorphism
    poly = [v for v in range(12) if 0 < hap[:, v].sum() < hap.shape[0]]
    assert len(df) == len(poly) * (len(poly) - 1) // 2
    got = {(int(a), int(b)): (float(r), float(r2)) for a, b, r, r2 in
           zip(df["gidx_a"], df["gidx_b"], df["R_PHASED"], df["R2_PHASED"])}
    # the output schema stores every statistic as float32 (_empty_pairs), so
    # ~7 decimal digits is the precision floor of the FORMAT, not of the
    # arithmetic -- which is computed in float64 and asserted separately below
    for (i, j), (r, r2) in got.items():
        want = oracle_phased_r(hap, i, j)
        assert r == pytest.approx(want, rel=1e-6), f"pair {(i, j)}"
        assert r2 == pytest.approx(want ** 2, rel=1e-6), f"pair {(i, j)}"


def test_phased_arithmetic_is_float64_exact(tmp_path):
    """The kernel itself matches the oracle far tighter than float32 output."""
    hap = simulate_phased(40, 8, seed=5)
    pairs = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)],
                     dtype=np.int64)
    res = L.phased_from_haplotypes(hap.T, pairs)
    for k, (i, j) in enumerate(pairs):
        want = oracle_phased_r(hap, i, j)
        if np.isnan(want):
            assert np.isnan(res["r_phased"][k])
        else:
            assert res["r_phased"][k] == pytest.approx(want, abs=1e-14)


def test_phased_d_equals_pab_minus_papb(tmp_path):
    """D from the same haplotype counts, checked against the definition."""
    hap = simulate_phased(30, 6, seed=7)
    pairs = np.array([(i, j) for i in range(6) for j in range(i + 1, 6)],
                     dtype=np.int64)
    res = L.phased_from_haplotypes(hap.T, pairs)
    H = hap.shape[0]
    for k, (i, j) in enumerate(pairs):
        pA, pB = res["pA"][k], res["pB"][k]
        want_r = oracle_phased_r(hap, i, j)
        if np.isnan(want_r):
            continue
        # r * sqrt(pA qA pB qB) must reproduce D exactly
        d_from_r = res["r_phased"][k] * np.sqrt(pA * (1 - pA) * pB * (1 - pB))
        assert d_from_r == pytest.approx(oracle_phased_d(hap, i, j), abs=1e-14)
        assert res["n"][k] == H


def test_dosage_stats_on_a_phased_file_are_refused(tmp_path):
    """hap2bit and 2bit share bytes but not meaning, so "r" on a phased file
    must raise rather than return a plausible wrong number."""
    hap = simulate_phased(20, 5, seed=1)
    p = tmp_path / "phased.cugen"
    write_cugen_phased(str(p), hap)
    with pytest.raises(ValueError, match="phased"):
        L.ld_matrix(str(p), stats=("r", "r2"), **CPU)


def test_phased_stats_on_an_unphased_file_are_refused(tmp_path):
    """A 2bit file carries no phase, so r_phased cannot be served from it."""
    dos = simulate_haplotypes(20, 5, seed=1)
    p = tmp_path / "unphased.cugen"
    write_cugen(str(p), dos.T)
    with pytest.raises(ValueError, match="phase"):
        L.ld_matrix(str(p), stats=("r_phased",), **CPU)


def oracle_phased_dprime(hap, i, j):
    """D' = D / Dmax over haplotypes (Lewontin 1964), computed directly."""
    x = hap[:, i].astype(np.float64)
    y = hap[:, j].astype(np.float64)
    pA, pB = x.mean(), y.mean()
    qA, qB = 1 - pA, 1 - pB
    if min(pA, pB, qA, qB) <= 0:
        return np.nan
    D = (x * y).mean() - pA * pB
    dmax = min(pA * qB, qA * pB) if D > 0 else min(pA * pB, qA * qB)
    return float(D / dmax) if dmax > 0 else np.nan


def test_d_and_dprime_phased_match_the_definition(tmp_path):
    """d_phased/dp_phased come straight from haplotype counts -- no cubic."""
    hap = simulate_phased(35, 10, seed=11)
    p = tmp_path / "phased.cugen"
    write_cugen_phased(str(p), hap)

    df = L.ld_matrix(str(p), stats=("r_phased", "d_phased", "dp_phased"),
                     min_r2=0.0, min_obs=1, **CPU)

    got = {(int(a), int(b)): (float(d), float(dp)) for a, b, d, dp in
           zip(df["gidx_a"], df["gidx_b"], df["D_PHASED"], df["DP_PHASED"])}
    assert got, "no pairs returned"
    for (i, j), (d, dp) in got.items():
        assert d == pytest.approx(oracle_phased_d(hap, i, j), rel=1e-5,
                                  abs=1e-9), f"D pair {(i, j)}"
        want_dp = oracle_phased_dprime(hap, i, j)
        if not np.isnan(want_dp):
            assert dp == pytest.approx(want_dp, rel=1e-5), f"D' pair {(i, j)}"


def test_dprime_phased_is_bounded(tmp_path):
    """|D'| <= 1 by construction; a violation means Dmax is wrong."""
    hap = simulate_phased(35, 14, seed=13)
    p = tmp_path / "phased.cugen"
    write_cugen_phased(str(p), hap)
    df = L.ld_matrix(str(p), stats=("r_phased", "dp_phased"), min_r2=0.0,
                     min_obs=1, **CPU)
    dp = df["DP_PHASED"].to_numpy()
    assert np.all(np.abs(dp[np.isfinite(dp)]) <= 1.0 + 1e-6)


def test_r2_phased_em_matches_the_em_oracle(tmp_path):
    """On UNPHASED input, r2_phased_em is D_em^2/(pA qA pB qB).

    The oracle is the Excoffier & Slatkin (1995) EM already used by
    tests/test_ld.py -- a different algorithm from the production cubic, so
    agreement is evidence rather than tautology.
    """
    from test_ld import em_p_ab

    dos = simulate_haplotypes(60, 8, seed=17)
    p = tmp_path / "unphased.cugen"
    write_cugen(str(p), dos.T)

    df = L.ld_matrix(str(p), stats=("r", "r2", "r2_phased_em"), min_r2=0.0,
                     min_obs=1, sign_reference="alt", **CPU)
    assert "R2_PHASED_EM" in df.columns

    pairs = np.array([(i, j) for i in range(8) for j in range(i + 1, 8)],
                     dtype=np.int64)
    tabs = L.contingency_tables(dos, pairs)
    got = {(int(a), int(b)): float(v) for a, b, v in
           zip(df["gidx_a"], df["gidx_b"], df["R2_PHASED_EM"])}
    for k, (i, j) in enumerate(pairs):
        if (i, j) not in got:
            continue
        tab = tabs[k]
        n = tab.sum()
        rows, cols = tab.sum(axis=1), tab.sum(axis=0)
        pA = (rows[1] + 2 * rows[2]) / (2 * n)
        pB = (cols[1] + 2 * cols[2]) / (2 * n)
        qA, qB = 1 - pA, 1 - pB
        if min(pA, pB, qA, qB) <= 0:
            continue
        D = em_p_ab(tab) - pA * pB
        want = D * D / (pA * qA * pB * qB)
        assert got[(i, j)] == pytest.approx(want, rel=2e-3, abs=1e-6), f"pair {(i, j)}"


@requires_cudf
def test_gpu_fused_phased_matches_numpy_reference(tmp_path, capsys):
    """The fused GPU path and the NumPy reference must agree on phased r.

    Both count haplotypes exactly, so this is an equality check up to float32
    output, not a tolerance negotiation. The capsys assertion is load-bearing:
    a parity test that silently routed around the fused kernel would be
    checking nothing (cf. tests/test_ld.py::test_fused_path_matches_plink2_golden).
    """
    hap = simulate_phased(300, 400, seed=23)
    p = tmp_path / "phased.cugen"
    write_cugen_phased(str(p), hap)

    ref = L.ld_matrix(str(p), stats=("r_phased", "r2_phased"), min_r2=0.0,
                      min_obs=1, backend="numpy", verbose=False)
    got = L.ld_matrix(str(p), stats=("r_phased", "r2_phased"), min_r2=0.2,
                      output=str(tmp_path / "out.tsv"), backend="gpu",
                      verbose=True)
    assert "fused kernel" in capsys.readouterr().out, \
        "fused path was not taken -- this test would be checking nothing"

    # the fused path returns a cudf.DataFrame, whose Series are deliberately
    # not iterable -- go through .to_numpy() so this works for either backend
    def as_map(df, col):
        return {(int(a), int(b)): float(v) for a, b, v in
                zip(df["gidx_a"].to_numpy(), df["gidx_b"].to_numpy(),
                    df[col].to_numpy())}

    r_ref = as_map(ref, "R_PHASED")
    r_got = as_map(got, "R_PHASED")
    assert r_got, "GPU path returned nothing"
    for k, v in r_got.items():
        assert k in r_ref, f"GPU emitted a pair the reference did not: {k}"
        assert v == pytest.approx(r_ref[k], rel=1e-6, abs=1e-7), f"pair {k}"
    # every reference pair clearing the threshold must be present
    want = {k for k, v in r_ref.items() if v * v >= 0.2}
    assert want <= set(r_got), f"GPU dropped {len(want - set(r_got))} pairs"


@requires_gpu
def test_hap_plane_is_the_documented_bit_order(tmp_path):
    """_build_h must read haplotype j from byte j>>3, bit 7-(j&7).

    Getting this wrong swaps the two haplotypes of each sample, which leaves
    allele frequencies IDENTICAL and only perturbs two-locus terms -- the
    hardest kind of error to notice downstream.
    """
    import cupy as cp
    hap = simulate_phased(8, 3, seed=29)          # 16 haplotypes, 3 variants
    p = tmp_path / "phased.cugen"
    write_cugen_phased(str(p), hap)
    reader = L.read_cugen(str(p))
    packed = cp.asarray(
        np.frombuffer(reader.read_packed_bytes(), dtype=np.uint8)
    ).reshape(3, int(reader.bytes_per_variant))
    plane = L._build_h(packed, 0, 3, 16, int(reader.bytes_per_variant))
    np.testing.assert_array_equal(cp.asnumpy(plane), hap.T.astype(np.float32))
