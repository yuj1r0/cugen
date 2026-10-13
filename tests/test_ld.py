"""Tests for cugen.ld.

Most of these run on CPU via backend="numpy"; the GPU set is marked and skips
automatically when CuPy is absent.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cugen import ld as L
from cugen.write import ENCODING_FLOAT32, write_cugen
from conftest import requires_cudf, requires_gpu, simulate_haplotypes

CPU = dict(backend="numpy", verbose=False)


# ---------------------------------------------------------------------------
# an independent EM oracle -- deliberately a DIFFERENT algorithm from the
# production cubic, so agreement is evidence rather than tautology.
# Excoffier & Slatkin (1995) Mol Biol Evol 12(5):921-927.
# ---------------------------------------------------------------------------
def em_p_ab(tab, n_iter=5000, n_starts=5):
    tab = np.asarray(tab, dtype=np.float64)
    n = tab.sum()
    rows, cols = tab.sum(axis=1), tab.sum(axis=0)
    pA = (rows[1] + 2 * rows[2]) / (2 * n)
    pB = (cols[1] + 2 * cols[2]) / (2 * n)
    if min(pA, pB) <= 1e-12 or max(pA, pB) >= 1 - 1e-12:
        return pA * pB
    c = 2 * tab[2, 2] + tab[2, 1] + tab[1, 2]
    n11 = tab[1, 1]
    lo, hi = max(0.0, pA + pB - 1.0), min(pA, pB)
    best, best_ll = lo, -np.inf
    for s in range(n_starts):
        x = lo + (hi - lo) * (s + 0.5) / n_starts
        for _ in range(n_iter):
            y, z, w = pA - x, pB - x, 1 - pA - pB + x
            den = x * w + y * z
            ww = (x * w) / den if den > 1e-12 else 0.0
            # a COUPLING double het yields ONE AB haplotype, not two
            new = min(max((c + n11 * ww) / (2 * n), lo), hi)
            if abs(new - x) < 1e-15:
                x = new
                break
            x = new
        ll = L._loglik(np.array(x), tab[None], np.array(pA), np.array(pB))[0]
        if ll > best_ll:
            best, best_ll = x, ll
    return best


def _pairs_index(df):
    return set(zip(df["gidx_a"].tolist(), df["gidx_b"].tolist()))


# --------------------------------------------------------------------- core
def test_r_matches_corrcoef(small_cugen):
    path, dos = small_cugen
    df = L.ld_matrix(path, **CPU)
    g = dos.astype(float)
    for _, row in df.iterrows():
        a, b = int(row["gidx_a"]), int(row["gidx_b"])
        ref = np.corrcoef(g[a], g[b])[0, 1]
        assert abs(row["R"] - ref) < 1e-5, (a, b, row["R"], ref)


def test_r2_internal_consistency(small_cugen):
    df = L.ld_matrix(small_cugen[0], **CPU)
    assert np.allclose(df["R2"], df["R"] ** 2, atol=1e-6)
    np.testing.assert_allclose(df["R2_SIGNED"], np.sign(df["R"]) * df["R2"],
                               atol=1e-7)
    assert (df["R2"] >= 0).all() and (df["R2"] <= 1).all()
    assert (df["DP"].abs() <= 1 + 1e-6).all()


def test_cubic_matches_em_oracle():
    """Production cubic vs an independent multi-start EM on random tables."""
    rng = np.random.default_rng(4)
    tabs = rng.integers(0, 50, size=(3000, 3, 3)).astype(float)
    tabs = tabs[tabs.sum(axis=(1, 2)) >= 10]
    out = L.ld_from_counts(tabs, dprime_method="phased")
    worst = 0.0
    for i in range(len(tabs)):
        t = tabs[i]
        n = t.sum()
        rows, cols = t.sum(1), t.sum(0)
        pA = (rows[1] + 2 * rows[2]) / (2 * n)
        pB = (cols[1] + 2 * cols[2]) / (2 * n)
        d_em = em_p_ab(t) - pA * pB
        if np.isfinite(out["d"][i]):
            worst = max(worst, abs(out["d"][i] - d_em))
    assert worst < 1e-8, f"cubic and EM disagree by {worst:.3e}"


def test_dprime_is_one_for_perfect_ld():
    """Two haplotypes only -> D' is exactly 1. The one analytic case."""
    tab = np.zeros((1, 3, 3))
    tab[0, 0, 0], tab[0, 1, 1], tab[0, 2, 2] = 25, 50, 25
    out = L.ld_from_counts(tab)
    assert abs(out["dp"][0] - 1.0) < 1e-9
    assert abs(out["r"][0] - 1.0) < 1e-9


def test_composite_d_matches_closed_form():
    rng = np.random.default_rng(8)
    tabs = rng.integers(1, 40, size=(500, 3, 3)).astype(float)
    out = L.ld_from_counts(tabs, dprime_method="composite")
    expect = out["r"] * np.sqrt(out["pA"] * (1 - out["pA"])
                                * out["pB"] * (1 - out["pB"]))
    np.testing.assert_allclose(out["d"], expect, atol=1e-12)


def test_counts_are_exact_integers():
    """The whole design rests on the contingency table being exact."""
    dos = simulate_haplotypes(300, 20, seed=3, missing_rate=0.12)
    pairs = np.array([(i, j) for i in range(20) for j in range(i + 1, 20)])
    tabs = L.contingency_tables(dos, pairs)
    for k, (i, j) in enumerate(pairs[:40]):
        both = (dos[i] != 3) & (dos[j] != 3)
        for a in range(3):
            for b in range(3):
                want = int(np.sum((dos[i][both] == a) & (dos[j][both] == b)))
                assert tabs[k, a, b] == want


# ------------------------------------------------------------- missingness
def test_pairwise_complete_case(missing_cugen):
    """N_OBS is the co-observed count, R uses only co-observed samples, and
    R is NOT the missing-as-dosage-0 value (the read_to_gpu trap, 85ff1b0)."""
    path, dos = missing_cugen
    df = L.ld_matrix(path, **CPU)
    row = df[(df["gidx_a"] == 3) & (df["gidx_b"] == 7)].iloc[0]
    both = (dos[3] != 3) & (dos[7] != 3)
    assert int(row["N_OBS"]) == int(both.sum())
    ref = np.corrcoef(dos[3][both].astype(float), dos[7][both].astype(float))[0, 1]
    assert abs(row["R"] - ref) < 1e-5
    z0 = np.where(dos == 3, 0, dos).astype(float)
    wrong = np.corrcoef(z0[3], z0[7])[0, 1]
    assert abs(row["R"] - wrong) > 1e-3, "missing is being treated as dosage 0"


def test_n_obs_equals_n_samples_without_missing(small_cugen):
    df = L.ld_matrix(small_cugen[0], **CPU)
    assert (df["N_OBS"] == 200).all()


# ------------------------------------------------------------------ windows
def test_window_pairs_emitted_exactly_once(small_cugen):
    df = L.ld_matrix(small_cugen[0], window=3, **CPU)
    want = {(i, j) for i in range(12) for j in range(i + 1, 12) if j - i <= 3}
    assert _pairs_index(df) == want
    assert len(df) == len(want)
    assert (df["gidx_a"] < df["gidx_b"]).all()


def test_no_window_is_all_pairs(small_cugen):
    df = L.ld_matrix(small_cugen[0], **CPU)
    assert len(df) == 12 * 11 // 2


def test_variant_range_selects_rows(small_cugen):
    df = L.ld_matrix(small_cugen[0], variant_range=(2, 6), **CPU)
    assert _pairs_index(df) == {(i, j) for i in range(2, 6)
                                for j in range(i + 1, 6)}


def test_min_r2_is_monotone(small_cugen):
    lo = L.ld_matrix(small_cugen[0], min_r2=0.05, **CPU)
    hi = L.ld_matrix(small_cugen[0], min_r2=0.5, **CPU)
    assert _pairs_index(hi) <= _pairs_index(lo)
    assert (hi["R2"] >= 0.5).all()


# ------------------------------------------------------------------- errors
def test_window_kb_requires_annotation(small_cugen):
    with pytest.raises(ValueError, match="annotation"):
        L.ld_matrix(small_cugen[0], window_kb=10.0, **CPU)


def test_region_requires_annotation(small_cugen):
    with pytest.raises(ValueError, match="annotation"):
        L.ld_matrix(small_cugen[0], region="22:1-1000", **CPU)


def test_non_cugen_path_names_the_converter(tmp_path):
    p = tmp_path / "x.vcf.gz"
    p.write_bytes(b"")
    with pytest.raises(ValueError, match="vcf2cugen"):
        L.ld_matrix(str(p), **CPU)


def test_non_2bit_encoding_raises(tmp_path):
    dos = simulate_haplotypes(50, 4, seed=1).T.astype(np.float32)
    p = tmp_path / "f32.cugen"
    write_cugen(str(p), dos, encoding=ENCODING_FLOAT32)
    with pytest.raises(NotImplementedError, match="2-bit"):
        L.ld_matrix(str(p), **CPU)


def test_bad_arguments_rejected(small_cugen):
    path = small_cugen[0]
    with pytest.raises(ValueError, match="unknown stats"):
        L.ld_matrix(path, stats=("r", "nope"), **CPU)
    with pytest.raises(ValueError, match="dprime_method"):
        L.ld_matrix(path, dprime_method="em", **CPU)
    with pytest.raises(ValueError, match="sign_reference"):
        L.ld_matrix(path, sign_reference="ref", **CPU)
    with pytest.raises(NotImplementedError, match="pairwise"):
        L.ld_matrix(path, missing="impute", **CPU)


def test_max_pairs_guard_raises_before_decode(small_cugen, monkeypatch):
    called = []
    monkeypatch.setattr(L, "contingency_tables",
                        lambda *a, **k: called.append(1))
    with pytest.raises(ValueError, match="window"):
        L.ld_matrix(small_cugen[0], max_pairs=5, **CPU)
    assert not called, "guard must fire before any genotype decode"


def test_unsorted_positions_raise():
    with pytest.raises(ValueError, match="non-decreasing"):
        L._plan_pairs(4, np.array([10, 5, 20, 30]), None, 1.0)


# ------------------------------------------------------------------- schema
def test_empty_result_keeps_full_schema(small_cugen):
    df = L.ld_matrix(small_cugen[0], min_r2=1.01, **CPU)
    assert len(df) == 0
    assert list(df.columns) == list(L._empty_pairs(L._DEFAULT_STATS).columns)
    assert df["N_OBS"].dtype == np.int32
    assert df["R"].dtype == np.float32


def test_stats_subset_controls_columns(small_cugen):
    df = L.ld_matrix(small_cugen[0], stats=("r", "r2"), **CPU)
    assert "D" not in df.columns and "DP" not in df.columns
    assert "R" in df.columns and "R2" in df.columns


def test_annotation_absent_placeholders(small_cugen):
    df = L.ld_matrix(small_cugen[0], **CPU)
    assert (df["CHR_A"] == 22).all()          # parsed from chr22.cugen
    assert (df["POS_A"] == 0).all()
    assert (df["ID_A"] == ".").all()


def test_output_roundtrip(small_cugen, tmp_path):
    out = tmp_path / "o.tsv"
    df = L.ld_matrix(small_cugen[0], output=str(out), **CPU)
    back = pd.read_csv(out, sep="\t")
    assert len(back) == len(df)
    np.testing.assert_allclose(back["R2"], df["R2"], atol=1e-6)


# -------------------------------------------------------------- orientation
def test_sign_reference_major_flips_exactly_the_right_pairs(small_cugen):
    path, dos = small_cugen
    a = L.ld_matrix(path, sign_reference="alt", **CPU)
    m = L.ld_matrix(path, sign_reference="major", **CPU)
    af = (dos.astype(float).sum(1)) / (2 * dos.shape[1])
    flip = np.array([(af[int(x)] > 0.5) ^ (af[int(y)] > 0.5)
                     for x, y in zip(a["gidx_a"], a["gidx_b"])])
    assert flip.any(), "fixture never exercises the orientation difference"
    np.testing.assert_allclose(m["R"], np.where(flip, -a["R"], a["R"]), atol=1e-6)
    np.testing.assert_allclose(m["R2"], a["R2"], atol=1e-6)


# --------------------------------------------------------------- contracts
def test_kernel_source_is_pure_ascii():
    """Regression guard for 34b4a59 (NVRTC crash on a non-ASCII char)."""
    assert L._LD_PLANES_SRC.isascii()


def test_cpu_only_import_and_error(monkeypatch, small_cugen):
    import cugen
    import cugen.ld
    monkeypatch.setattr(L, "HAS_CUPY", False)
    with pytest.raises(RuntimeError, match="CuPy"):
        L.ld_matrix(small_cugen[0], backend="gpu", verbose=False)


def test_alias_is_wired():
    import cugen as cg
    assert cg.r2 is cg.ld_matrix


def test_ld_clump_is_implemented_and_aliased():
    """Was `test_ld_clump_still_stubbed`. ld_clump is real now, so the alias
    _stubs.py used to intercept must resolve to the implementation, and the
    call must fail on its own signature rather than on NotImplementedError."""
    import cugen as cg
    assert cg.clump is L.ld_clump
    with pytest.raises(TypeError):          # cugen and sumstats are required
        L.ld_clump()


# --------------------------------------------------------------------- gpu
@requires_gpu
def test_gpu_matches_numpy_backend(small_cugen):
    path, _ = small_cugen
    a = L.ld_matrix(path, backend="numpy", verbose=False)
    b = L.ld_matrix(path, backend="gpu", verbose=False)
    pd.testing.assert_frame_equal(
        a.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True),
        b.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True),
        atol=1e-5, check_dtype=True)


@requires_gpu
def test_gpu_matches_numpy_with_missing(missing_cugen):
    path, _ = missing_cugen
    a = L.ld_matrix(path, backend="numpy", verbose=False)
    b = L.ld_matrix(path, backend="gpu", verbose=False)
    np.testing.assert_array_equal(a["N_OBS"], b["N_OBS"])
    np.testing.assert_allclose(a["R"], b["R"], atol=1e-5)


# ------------------------------------------------------------ plink2 parity
# Golden files are committed, so this asserts parity in CI forever with no
# plink installed. See tests/data/README.md for the exact regeneration
# commands and plink2 version.
DATA = Path(__file__).parent / "data"


def _read_vcor(path):
    rows = {}
    with open(path) as fh:
        hdr = fh.readline().lstrip("#").split()
        for line in fh:
            d = dict(zip(hdr, line.split()))
            rows[(d["ID_A"], d["ID_B"])] = d
    return rows


@pytest.mark.parametrize("stat,col,source", [
    ("R", "UNPHASED_R", "unphased"),
    ("D", "D", "phased"),
    ("DP", "DPRIME", "phased"),
])
def test_plink2_golden_parity(tmp_path, stat, col, source):
    dos = np.load(DATA / "ld_fixture.npy")
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), dos.T.astype(np.uint8))
    df = L.ld_matrix(str(path), sign_reference="major", **CPU)
    gold = _read_vcor(DATA / f"ld_fixture_plink2_{source}.vcor")

    got = {(f"v{int(a)}", f"v{int(b)}"): v
           for a, b, v in zip(df["gidx_a"], df["gidx_b"], df[stat])}
    checked, worst = 0, 0.0
    for key, row in gold.items():
        if key not in got:            # monomorphic variant 11 is dropped
            continue
        worst = max(worst, abs(got[key] - float(row[col])))
        checked += 1
    assert checked >= 50, f"only {checked} pairs compared"
    assert worst < 2e-6, f"{stat} vs plink2 {col}: max abs err {worst:.3e}"


def test_monomorphic_variant_is_dropped(tmp_path):
    dos = np.load(DATA / "ld_fixture.npy")
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), dos.T.astype(np.uint8))
    df = L.ld_matrix(str(path), **CPU)
    assert 11 not in set(df["gidx_a"]) | set(df["gidx_b"])


def test_plink2_n_obs_matches_pairwise_complete(tmp_path):
    dos = np.load(DATA / "ld_fixture.npy")
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), dos.T.astype(np.uint8))
    df = L.ld_matrix(str(path), **CPU)
    row = df[(df["gidx_a"] == 4) & (df["gidx_b"] == 9)].iloc[0]
    both = (dos[4] != 3) & (dos[9] != 3)
    assert int(row["N_OBS"]) == int(both.sum()) < 60


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("tile", [2, 3, 5, 7])
def test_gpu_tiling_is_boundary_exact(small_cugen, tile):
    """Tiny tiles force many block boundaries. The emitted pair SET and every
    statistic must be identical to the untiled NumPy reference -- off-by-one at
    a tile edge is the classic failure and a count check alone would miss it."""
    path, _ = small_cugen
    ref = L.ld_matrix(path, backend="numpy", verbose=False)
    got = L.ld_matrix(path, backend="gpu", tile_size=tile, verbose=False)
    assert _pairs_index(got) == _pairs_index(ref)
    a = ref.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    b = got.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    for col in ("R", "R2", "D", "DP", "N_OBS"):
        np.testing.assert_allclose(a[col], b[col], atol=1e-5, err_msg=col)


@requires_gpu
@pytest.mark.gpu
@pytest.mark.parametrize("tile", [2, 4])
def test_gpu_tiling_with_window_and_missing(missing_cugen, tile):
    path, _ = missing_cugen
    ref = L.ld_matrix(path, window=4, backend="numpy", verbose=False)
    got = L.ld_matrix(path, window=4, backend="gpu", tile_size=tile,
                      verbose=False)
    assert _pairs_index(got) == _pairs_index(ref)
    a = ref.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    b = got.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    np.testing.assert_array_equal(a["N_OBS"], b["N_OBS"])
    np.testing.assert_allclose(a["R"], b["R"], atol=1e-5)


def test_tile_plan_peak_is_bounded_in_p():
    """The scalability invariant, tested deterministically.

    Sampling peak device memory from a thread was flaky: the workloads small
    enough for a unit test finish before the sampler observes anything, and
    reading pool.total_bytes() afterwards is not a high-water mark because
    ld_matrix frees as it goes. So assert the property the design actually
    guarantees -- the per-tile working set is a function of tile size and
    sample count, never of the variant count -- and leave the empirical
    plateau to the chr22 benchmark, which has the scale to show it.
    """
    def predicted_bytes(B, n_samples, n_metrics=9):
        planes = 2 * 3 * B * n_samples * 4          # two blocks, three planes
        counts = n_metrics * B * B * 4
        return planes + counts

    B, ns = 2048, 2504
    at = [predicted_bytes(B, ns) for _ in (10_000, 100_000, 1_000_000)]
    assert len(set(at)) == 1, "per-tile working set must not depend on p"

    # and the planner must not grow the tile without bound
    assert L._tile_size_for.__doc__ is not None
    for w in (100, 500, 5000, None):
        # emulate the clamp without needing a device
        B0 = 8192
        wb = None if w is None else max(256, ((w + 255) // 256) * 256)
        got = B0 if wb is None else min(B0, wb)
        assert 256 <= got <= 8192
        if w is not None and w < 8192:
            assert got <= wb, "tile must shrink to the window"


def test_cubic_picks_the_global_maximum_likelihood_root():
    """Real 1000 Genomes table where the likelihood has three admissible roots.

    Our solver picks the global maximum (verified here against a dense brute
    force over the admissible interval). plink2 selects a different root on
    tables like this -- 44 of 854,850 pairs (0.005%) on chr22 -- so D/D' can
    disagree with plink2 even though r and r^2 match exactly. Documented in
    the module docstring; this test pins OUR behaviour, which is the ML.
    """
    tab = np.array([[0, 0, 0], [2, 632, 31], [14, 1705, 120]], float)
    n = tab.sum()
    rows, cols = tab.sum(1), tab.sum(0)
    pA = (rows[1] + 2 * rows[2]) / (2 * n)
    pB = (cols[1] + 2 * cols[2]) / (2 * n)
    lo, hi = max(0.0, pA + pB - 1.0), min(pA, pB)

    xs = np.linspace(lo, hi, 200001)
    lls = L._loglik(xs, np.repeat(tab[None], len(xs), 0),
                    np.full(len(xs), pA), np.full(len(xs), pB))
    x_brute = xs[np.argmax(lls)]

    out = L.ld_from_counts(tab[None])
    d_brute = x_brute - pA * pB
    assert abs(out["d"][0] - d_brute) < 1e-5, (out["d"][0], d_brute)
    # and the ML root here is NOT the one nearest the composite estimate,
    # which is exactly why a naive "closest root" rule diverges
    assert out["d"][0] < 0 < out["r"][0]


@requires_gpu
@pytest.mark.gpu
def test_fused_kernel_matches_reference(tmp_path):
    """The fused single-kernel path must agree exactly with the tiled path.

    It bypasses the whole CuPy epilogue -- r is computed inside the kernel in
    double, survivors are compacted with an atomic counter, and nothing syncs
    per tile. That is a lot of machinery to get subtly wrong, so compare the
    emitted pair SET and every value against backend='numpy'.
    """
    dos = simulate_haplotypes(300, 900, seed=17, missing_rate=0.0)
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), dos.T.astype(np.uint8))
    out = tmp_path / "o.tsv"

    ref = L.ld_matrix(str(path), window=50, min_r2=0.1, stats=("r", "r2"),
                      backend="numpy", verbose=False)
    got = L.ld_matrix(str(path), window=50, min_r2=0.1, stats=("r", "r2"),
                      backend="gpu", output=str(out), verbose=False)
    got = got.to_pandas() if hasattr(got, "to_pandas") else got

    assert _pairs_index(got) == _pairs_index(ref)
    a = ref.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    b = got.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    np.testing.assert_allclose(a["R"], b["R"], atol=1e-5)
    np.testing.assert_allclose(a["R2"], b["R2"], atol=1e-5)
    assert out.exists() and out.stat().st_size > 0


@requires_gpu
@pytest.mark.gpu
def test_fused_kernel_survives_buffer_overflow(tmp_path):
    """Undersized output buffer must be detected and retried, not truncated."""
    dos = simulate_haplotypes(200, 600, seed=5, missing_rate=0.0)
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), dos.T.astype(np.uint8))
    ref = L.ld_matrix(str(path), min_r2=0.0, stats=("r",), backend="numpy",
                      verbose=False)
    got = L.ld_matrix(str(path), min_r2=0.0, stats=("r",), backend="gpu",
                      output=str(tmp_path / "o.tsv"), max_pairs=10**12,
                      verbose=False)
    got = got.to_pandas() if hasattr(got, "to_pandas") else got
    assert len(got) == len(ref), (len(got), len(ref))


def test_precision_rejects_half_with_an_explanation():
    """fp16 must be refused, not silently allowed.

    Same 10-bit mantissa as TF32 and the inputs {0,1,2} are exactly
    representable, so it looks safe -- but cuBLAS accumulates half matmuls in
    half, so integer exactness ends at 2**11 and the accumulator saturates at
    65,504. Measured max|err| = inf at n >= 100,000: silent at small n,
    catastrophic at biobank scale.
    """
    for bad in ("fp16", "half", "bf16"):
        with pytest.raises(ValueError, match="half"):
            L._resolve_precision(bad, 100_000, False)
    with pytest.raises(ValueError, match="must be"):
        L._resolve_precision("float8", 1000, False)


def test_precision_fp32_never_uses_tf32():
    assert L._resolve_precision("fp32", 100_000, False) is False


@requires_gpu
@pytest.mark.gpu
def test_tf32_is_bit_identical_to_fp32(tmp_path):
    """TF32 must change speed and nothing else.

    This is the whole justification for defaulting it on: the plane values
    need two mantissa bits, TF32 has ten, and it accumulates in fp32.
    """
    dos = simulate_haplotypes(2000, 600, seed=23, missing_rate=0.0)
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), dos.T.astype(np.uint8))
    kw = dict(window=60, min_r2=0.05, stats=("r", "r2"), backend="gpu",
              verbose=False)
    a = L.ld_matrix(str(path), precision="fp32",
                    output=str(tmp_path / "a.tsv"), **kw)
    b = L.ld_matrix(str(path), precision="tf32",
                    output=str(tmp_path / "b.tsv"), **kw)
    a = a.to_pandas() if hasattr(a, "to_pandas") else a
    b = b.to_pandas() if hasattr(b, "to_pandas") else b
    assert len(a) == len(b) and len(a) > 100
    a = a.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    b = b.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    np.testing.assert_array_equal(a["R"].to_numpy(), b["R"].to_numpy())


@requires_cudf
@pytest.mark.gpu
def test_fused_path_matches_plink2_golden(tmp_path, capsys):
    """plink2 parity ON THE FUSED PATH, and proof that path was taken.

    The first parity check after the optimisation work passed while silently
    exercising the table path: compare_plink.py requests D/D' and no output
    file, both of which force need_table and route around the fused kernel,
    cuDF assembly and the r-only moment formula. A green result on the wrong
    path is worse than no check, so assert the path explicitly.
    """
    dos = np.load(DATA / "ld_fixture.npy")
    clean = dos.copy()
    clean[clean == 3] = 0          # fused path requires no missingness
    clean = np.delete(clean, 11, axis=0)      # drop the monomorphic variant
    path = tmp_path / "chr22.cugen"
    write_cugen(str(path), clean.T.astype(np.uint8))

    ref = L.ld_matrix(str(path), stats=("r", "r2"), min_r2=0.0,
                      backend="numpy", verbose=False)
    got = L.ld_matrix(str(path), stats=("r", "r2"), min_r2=0.0,
                      backend="gpu", output=str(tmp_path / "o.tsv"),
                      max_pairs=10 ** 12, verbose=True)
    assert "fused kernel" in capsys.readouterr().out, \
        "fused path was not taken -- this test would be checking nothing"

    got = got.to_pandas() if hasattr(got, "to_pandas") else got
    a = ref.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    b = got.sort_values(["gidx_a", "gidx_b"]).reset_index(drop=True)
    assert len(a) == len(b) and len(a) > 20
    np.testing.assert_allclose(a["R"], b["R"], atol=1e-5)
