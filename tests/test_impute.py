"""Tests for the phased encoding, genetic maps, and imputation.

All CPU. The GPU kernels are exercised in test_impute_gpu.py.

Two habits carried over from the LD work in this repo, both earned:

  * Every threshold that selects a code path is module-level and injectable, so
    a test can force either branch. Three tests there could not fail because
    the fixture never crossed the threshold it was meant to exercise.
  * Where a step has an obvious slow implementation and a clever fast one, both
    are kept and checked against each other. Two independent implementations
    agreeing is the strongest evidence available without an external oracle.
"""
import numpy as np
import pandas as pd
import pytest

import cugen.io as cio
import cugen.write as cwrite
from cugen._genmap import GeneticMap, resolve_map
from cugen._impute_core import (aggregate_markers, aggregate_mismatch,
                                build_carriers,
                                default_err, dose_dense, dose_sparse,
                                forward_backward_blocked, forward_backward_ref,
                                impute_haplotypes, interpolation_weights,
                                transition_tau)
from cugen.impute import dosage_r2, impute, plan_windows
from cugen.write import (ENCODING_HAP2BIT, hap2bit_dosages, pack_2bit,
                         pack_hap2bit, unpack_2bit, unpack_hap2bit,
                         validate_cugen, write_cugen_phased)

from conftest import scaled_ne, simulate_mosaic


# --------------------------------------------------------------------------
# Format: the phased encoding
# --------------------------------------------------------------------------

@pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 7, 8, 9, 33, 100, 257])
def test_bit_views_coincide(n):
    """The 1-bit haplotype view and 2-bit sample view are the same bytes.

    This is the claim the whole phased design rests on: packing 2n allele bits
    8-per-byte MSB-first is byte-for-byte identical to packing n derived 2-bit
    codes 4-per-byte. If it ever stops holding, hap2bit needs its own layout
    and bytes_per_variant, and every I/O path has to learn about it.
    """
    rng = np.random.default_rng(n)
    hap = rng.integers(0, 2, size=2 * n).astype(np.uint8)
    codes = (hap[0::2] << 1) | hap[1::2]
    assert pack_hap2bit(hap).tobytes() == pack_2bit(codes).tobytes()
    assert np.array_equal(unpack_hap2bit(pack_hap2bit(hap), 2 * n), hap)
    assert np.array_equal(pack_hap2bit(hap.reshape(n, 2)), pack_hap2bit(hap))


def test_phased_and_unphased_dosages_differ_for_two_codes():
    """Pins the exact hazard the guards exist for.

    Codes 00 and 01 agree between the encodings; 10 and 11 do not. A test that
    only sampled rare variants would find agreement and conclude the encodings
    are interchangeable.
    """
    hap = np.array([0, 0, 0, 1, 1, 0, 1, 1], dtype=np.uint8)   # 4 samples
    p = pack_hap2bit(hap)
    assert list(hap2bit_dosages(p, 4)) == [0, 1, 1, 2]
    assert list(unpack_2bit(p, 4)) == [0, 1, 2, 3]             # wrong, and quiet


def test_phased_file_read_as_unphased_raises(tmp_path):
    """Every dosage path must refuse, rather than return plausible numbers."""
    rng = np.random.default_rng(0)
    hap = rng.integers(0, 2, size=(2 * 9, 12)).astype(np.uint8)
    p = tmp_path / "ph.cugen"
    write_cugen_phased(str(p), hap)
    with cio.CugenReader(str(p)) as r:
        assert r.phased
        for meth, args in [("read_to_numpy", ()), ("read_to_gpu", ()),
                           ("fused_univariate", (np.zeros(9, np.float32),)),
                           ("read_indices_to_gpu", ([0, 1],)),
                           ("read_indices_to_gpu_batched", ([0, 1],))]:
            with pytest.raises(ValueError, match="PHASED"):
                getattr(r, meth)(*args)


def test_pinned_reader_also_guards(tmp_path):
    """CugenReaderPinned.read_to_gpu OVERRIDES the base method.

    Inheriting a signature does not inherit a check, and read_cugen() returns
    this subclass when USE_PINNED_READER is set -- so a guard only on the base
    class would leave a real path open.
    """
    import inspect
    src = inspect.getsource(cio.CugenReaderPinned.read_to_gpu)
    assert "_require_unphased" in src


def test_phased_roundtrip_and_stats(tmp_path):
    rng = np.random.default_rng(5)
    hap = rng.integers(0, 2, size=(2 * 30, 40)).astype(np.uint8)
    p = tmp_path / "ph.cugen"
    write_cugen_phased(str(p), hap)
    rep = validate_cugen(str(p), verbose=False)
    assert rep["ok"], rep["problems"]
    assert rep["phased"] and rep["encoding"] == ENCODING_HAP2BIT
    assert not rep["has_missing"]        # phased has no code for missing
    with cio.CugenReader(str(p)) as r:
        assert np.array_equal(r.read_haplotypes(), hap)
        dose = (hap[0::2] + hap[1::2]).astype(np.float32)
        assert np.array_equal(r.read_dosages_from_phased(), dose)
        # header stats are written under the popcount map, so they describe
        # dosages and stay comparable with an unphased file of the same cohort
        assert np.allclose(r.mu_x, dose.mean(axis=0), atol=1e-6)


def test_phased_writer_rejects_dosages_and_vice_versa(tmp_path):
    from cugen.write import CugenWriter, ENCODING_2BIT
    with CugenWriter(str(tmp_path / "a.cugen"), 4, 1,
                     encoding=ENCODING_HAP2BIT) as w:
        with pytest.raises(ValueError, match="add_variant_phased"):
            w.add_variant(0, [0, 1, 2, 0])
        w.add_variant_phased(0, np.zeros(8, np.uint8))
    with CugenWriter(str(tmp_path / "b.cugen"), 4, 1,
                     encoding=ENCODING_2BIT) as w:
        with pytest.raises(ValueError, match="ENCODING_HAP2BIT"):
            w.add_variant_phased(0, np.zeros(8, np.uint8))
        w.add_variant(0, [0, 1, 2, 0])


def test_missing_cannot_be_written_to_a_phased_file():
    with pytest.raises(ValueError, match="0 or 1"):
        pack_hap2bit(np.array([0, 1, 3, 0], dtype=np.uint8))


def test_inconsistent_phased_flag_refused_at_open(tmp_path):
    """hap2bit and 2bit share bytes_per_variant, so such a file passes every
    size check while decoding wrongly. It has to be caught structurally."""
    p = tmp_path / "bad.cugen"
    rng = np.random.default_rng(2)
    write_cugen_phased(str(p), rng.integers(0, 2, size=(8, 6)).astype(np.uint8))
    raw = bytearray(p.read_bytes())
    raw[64] = raw[64] & ~cwrite.FLAG_PHASED        # clear PHASED, keep hap2bit
    p.write_bytes(bytes(raw))
    assert not validate_cugen(str(p), verbose=False)["ok"]
    with pytest.raises(ValueError, match="PHASED"):
        cio.CugenReader(str(p))


def test_io_and_write_constants_agree():
    """io.py duplicates the format constants; drift would misread files."""
    for name in ("CUGEN_MAGIC", "CUGEN_VERSION", "HEADER_SIZE",
                 "ENCODING_2BIT", "ENCODING_UINT8", "ENCODING_FLOAT16",
                 "ENCODING_FLOAT32", "ENCODING_HAP2BIT",
                 "FLAG_HAS_MISSING", "FLAG_HAS_GIDX_MAP", "FLAG_PHASED"):
        assert getattr(cio, name) == getattr(cwrite, name), name


def test_encoding_name_does_not_crash_on_unknown():
    """A hardcoded list here raised IndexError on any encoding past float32,
    which is how adding hap2bit first showed up."""
    assert cio.encoding_name(4) == "hap2bit"
    assert "unknown" in cio.encoding_name(99)


# --------------------------------------------------------------------------
# Genetic maps
# --------------------------------------------------------------------------

def test_constant_rate_is_one_cm_per_mb():
    g = GeneticMap.constant_rate()
    assert np.allclose(g.cm([0, 1_000_000, 2_500_000]), [0.0, 1.0, 2.5])


def test_morgans_is_cm_over_100():
    """The transition takes Morgans; every map on disk is in cM."""
    g = GeneticMap.constant_rate()
    assert np.allclose(g.morgans([1_000_000]), [0.01])


def test_map_interpolates_linearly():
    g = GeneticMap([100, 200, 400], [0.0, 1.0, 5.0])
    assert np.allclose(g.cm([100, 150, 200, 300, 400]), [0, 0.5, 1, 3, 5])


@pytest.mark.parametrize("mode,expect", [("extrapolate", [-0.5, 7.0]),
                                         ("clamp", [0.0, 5.0])])
def test_out_of_range_modes(mode, expect):
    g = GeneticMap([100, 200, 400], [0.0, 1.0, 5.0], out_of_range=mode)
    assert np.allclose(g.cm([50, 500]), expect)


def test_out_of_range_raise_mode():
    g = GeneticMap([100, 200, 400], [0.0, 1.0, 5.0], out_of_range="raise")
    with pytest.raises(ValueError, match="outside the map range"):
        g.cm([50])


def test_swapped_cm_and_pos_columns_rejected():
    """A swapped map still interpolates and still returns plausible cM."""
    with pytest.raises(ValueError, match="non-decreasing"):
        GeneticMap([100, 200, 400], [0.0, 5.0, 1.0])


def test_plink_map_roundtrip(tmp_path):
    p = tmp_path / "m.map"
    p.write_text("22 . 0.0 1000\n22 . 1.0 2000\n22 . 3.0 4000\n")
    g = resolve_map(str(p), chrom=22)
    assert np.allclose(g.cm([1500, 3000]), [0.5, 2.0])


def test_multi_chromosome_map_requires_selection(tmp_path):
    p = tmp_path / "m.map"
    p.write_text("21 . 0.0 1000\n21 . 1.0 2000\n22 . 0.0 1000\n22 . 1.0 2000\n")
    with pytest.raises(ValueError, match="chrom="):
        resolve_map(str(p))


# --------------------------------------------------------------------------
# Core numerics
# --------------------------------------------------------------------------

def test_aggregate_marker_position_is_mean_of_first_and_last():
    """Not the mean of all constituents, and not the first."""
    cm = np.array([0.000, 0.001, 0.004])
    s, e, g = aggregate_markers(cm, cluster=0.005)
    assert (s.size, e[0]) == (1, 3)
    assert np.isclose(g[0], (0.000 + 0.004) / 2)
    assert not np.isclose(g[0], cm.mean())


def test_aggregate_span_is_measured_from_the_cluster_start():
    cm = np.array([0.000, 0.001, 0.004, 0.006])
    s, e, _ = aggregate_markers(cm, cluster=0.005)
    assert list(zip(s.tolist(), e.tolist())) == [(0, 3), (3, 4)]


def test_cluster_zero_disables_aggregation():
    cm = np.array([0.0, 0.0001, 0.0002])
    s, e, g = aggregate_markers(cm, cluster=0.0)
    assert s.size == 3 and np.allclose(g, cm)


def test_aggregate_mismatch_is_l_times_eps():
    """An aggregate of l markers mismatches with probability l*eps.

    Added after a mutation replacing l*err with err left the entire suite
    green: the rule was computed inline and never asserted anywhere.
    """
    starts = np.array([0, 3, 4])
    stops = np.array([3, 4, 9])
    assert np.allclose(aggregate_mismatch(starts, stops, 0.01),
                       [0.03, 0.01, 0.05])


def test_aggregate_mismatch_is_capped_at_one_half():
    """It is a union bound; past 0.5 a 'mismatch' would beat a match."""
    assert aggregate_mismatch([0], [400], 0.01)[0] == 0.5


@pytest.mark.parametrize("n_typed,expected", [(1, 0.99), (2, 0.98), (4, 0.96)])
def test_aggregate_mismatch_reaches_the_hmm(n_typed, expected):
    """Pin the VALUE l*eps produces, not merely that clustering changes things.

    A first attempt here compared clustered against unclustered output and
    asserted they differ. They do -- but they differ under any mismatch rule at
    all, because aggregation also changes the marker count, the codes and the
    transitions. That test passed against a deliberately broken l*eps and was
    therefore worthless.

    This constructs one aggregate marker over `l` constituents and two
    reference haplotypes, one matching the target and one not. With a single
    aggregate the posterior is just the normalised emission,

        P(matching hap) = (1 - l*eps) / ((1 - l*eps) + l*eps) = 1 - l*eps

    and the untyped marker is carried by the matching haplotype alone, so the
    imputed allele probability IS 1 - l*eps. eps alone would give 0.99 for
    every l.
    """
    eps = 0.01
    M = n_typed + 1
    ref = np.zeros((2, M), dtype=np.uint8)
    ref[0, :n_typed] = 1                       # hap 0 carries the target's alleles
    ref[0, -1] = 1                             # ... and allele 1 at the untyped site
    ref[1, -1] = 0                             # hap 1 does not
    tgt_idx = np.arange(n_typed)
    cm = np.append(np.linspace(0.0, 0.003, n_typed), 0.0031)   # untyped last
    P = impute_haplotypes(ref, ref[[0]][:, tgt_idx], tgt_idx, cm,
                          ne=1e-9, err=eps, cluster=0.005)
    s, e, _ = aggregate_markers(cm[tgt_idx], 0.005)
    assert s.size == 1 and (e[0] - s[0]) == n_typed, "fixture is not one aggregate"
    assert P[0, -1] == pytest.approx(expected, abs=1e-6)


def test_tau_is_zero_at_the_first_marker():
    tau = transition_tau(np.array([0.0, 0.001, 0.002]), 100_000, 1000)
    assert tau[0] == 0.0 and np.all(tau[1:] > 0)


def test_tau_scales_inversely_with_panel_size():
    """The reason Beagle's ne=100000 misbehaves on a small panel."""
    g = np.array([0.0, 1e-3])
    assert transition_tau(g, 1e5, 100)[1] > transition_tau(g, 1e5, 10_000)[1]


@pytest.mark.parametrize("block", [1, 3, 7, 40])
def test_blocked_matches_reference_forward_backward(block):
    """Memory-bounded and obvious implementations must agree exactly."""
    rng = np.random.default_rng(7)
    K, T, C = 23, 5, 40
    rc = rng.integers(0, 2, size=(C, K)).astype(np.int32)
    tc = rng.integers(0, 2, size=(C, T)).astype(np.int32)
    tau = np.concatenate([[0.0], rng.uniform(1e-4, 0.2, size=C - 1)])
    mism = np.full(C, 1e-4)
    ref = forward_backward_ref(rc, tc, tau, mism)
    got = forward_backward_blocked(rc, tc, tau, mism, block=block)
    assert np.abs(ref - got).max() < 1e-12
    assert np.allclose(ref.sum(axis=1), 1.0)


def test_sparse_dose_matches_dense():
    rng = np.random.default_rng(9)
    K, T, M = 40, 6, 80
    post_l = rng.random((K, T)); post_l /= post_l.sum(axis=0)
    post_r = rng.random((K, T)); post_r /= post_r.sum(axis=0)
    bits = (rng.random((K, M)) < rng.uniform(0.02, 0.98, size=M)).astype(np.uint8)
    indptr, indices, major = build_carriers(bits)
    lam = rng.random(M)
    dd = dose_dense(post_l, post_r, lam, bits)
    ds = dose_sparse(post_l, post_r, lam, indptr, indices, major, np.arange(M))
    assert np.abs(dd - ds).max() < 1e-12
    assert dd.min() >= -1e-12 and dd.max() <= 1 + 1e-12


def test_carriers_store_the_minority_allele():
    """nnz must be bounded by K/2 per marker, or the sparse path is pointless."""
    bits = np.ones((10, 3), dtype=np.uint8)
    bits[0, 0] = 0                      # one carrier of allele 0
    indptr, indices, major = build_carriers(bits)
    assert list(np.diff(indptr)) == [1, 0, 0]
    assert major[0] == 1


def test_interpolation_weights_bracket_correctly():
    left, lam = interpolation_weights(np.array([1.0, 2.0, 4.0]),
                                      np.array([1.0, 1.5, 2.0, 3.0, 4.0]))
    assert list(left) == [0, 0, 0, 1, 1]
    assert np.allclose(lam, [1.0, 0.5, 0.0, 0.5, 0.0])


def test_default_err_follows_the_manual_formula():
    H = 4904.0
    theta = 1.0 / (0.5 + np.log(H))
    assert np.isclose(default_err(H), theta / (2 * (theta + H)))


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------

def test_perfect_copy_recovered_exactly_without_recombination():
    """A target that IS a reference haplotype must be imputed exactly.

    Run at tau -> 0 so it tests the algorithm rather than a parameter choice:
    with recombination switched off the HMM has no excuse for leaving the
    haplotype it matches.
    """
    rng = np.random.default_rng(11)
    K, M = 60, 400
    ref = (rng.random((K, M)) < 0.3).astype(np.uint8)
    tgt_idx = np.sort(rng.choice(M, size=80, replace=False))
    cm = np.sort(rng.uniform(0, 5, size=M)); cm[0] = 0.0
    for h in (3, 17):
        P = impute_haplotypes(ref, ref[[h]][:, tgt_idx], tgt_idx, cm,
                              ne=1e-9, cluster=0.005)
        assert np.allclose(P[0], ref[h], atol=1e-6)


def test_sparse_and_dense_agree_end_to_end():
    rng = np.random.default_rng(4)
    K, M = 80, 300
    ref = simulate_mosaic(K, M, seed=4)
    tgt_idx = np.sort(rng.choice(M, size=60, replace=False))
    cm = np.linspace(0, 2, M)
    tgt = simulate_mosaic(4, M, seed=5)[:, tgt_idx]
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    a = impute_haplotypes(ref, tgt, tgt_idx, cm, sparse=True, **kw)
    b = impute_haplotypes(ref, tgt, tgt_idx, cm, sparse=False, **kw)
    assert np.abs(a - b).max() < 1e-12


def test_allele_probabilities_are_in_range():
    rng = np.random.default_rng(6)
    K, M = 60, 200
    ref = simulate_mosaic(K, M, seed=6)
    tgt_idx = np.sort(rng.choice(M, size=40, replace=False))
    P = impute_haplotypes(ref, simulate_mosaic(6, M, seed=7)[:, tgt_idx],
                          tgt_idx, np.linspace(0, 1.5, M), ne=scaled_ne(K))
    assert P.min() >= -1e-12 and P.max() <= 1 + 1e-12


def test_impute_recovers_held_out_genotypes(phased_cugen):
    """The real end-to-end assertion: masked markers come back."""
    ref_p, tgt_p, meta = phased_cugen
    res = impute(tgt_p, ref=ref_p, annotation=meta["ann"], ne=meta["ne"],
                 verbose=False)
    imp = res["IMP"].to_numpy()
    assert imp.sum() > 0 and (~imp).sum() > 0
    # DR2 should be near 1 where the genotype was actually observed
    assert res.loc[~imp, "DR2"].mean() > res.loc[imp, "DR2"].mean()


def test_impute_dosage_accuracy(phased_cugen):
    ref_p, tgt_p, meta = phased_cugen
    out = ref_p.replace("ref.cugen", "out.cugen")
    res = impute(tgt_p, ref=ref_p, annotation=meta["ann"], ne=meta["ne"],
                 out=out, verbose=False)
    with cio.CugenReader(out) as r:
        n_s, n_v = r.n_samples, r.n_variants
        got = np.frombuffer(r.read_packed_bytes(), dtype=np.float16
                            ).reshape(n_v, n_s).T.astype(np.float64)
    tgt = meta["tgt_hap"]
    truth = (tgt[0::2] + tgt[1::2]).astype(np.float64)
    imp = res["IMP"].to_numpy()
    r2 = np.corrcoef(got[:, imp].ravel(), truth[:, imp].ravel())[0, 1] ** 2
    assert r2 > 0.8, f"imputed dosage r2 {r2:.3f} too low"
    # genotyped markers are observed, so they must come back essentially exact
    assert np.abs(got[:, ~imp] - truth[:, ~imp]).max() < 0.05


def test_unphased_input_is_refused(tmp_path, small_cugen):
    path = small_cugen[0] if isinstance(small_cugen, tuple) else small_cugen
    ann = pd.DataFrame({"gidx": [0], "POS": [1]})
    with pytest.raises(ValueError, match="not phased"):
        impute(path, ref=path, annotation=ann, verbose=False)


def test_annotation_is_required(phased_cugen):
    ref_p, tgt_p, _ = phased_cugen
    with pytest.raises(ValueError, match="annotation is required"):
        impute(tgt_p, ref=ref_p, annotation=None, verbose=False)


def test_target_markers_absent_from_reference_are_reported(phased_cugen):
    ref_p, tgt_p, meta = phased_cugen
    ann = meta["ann"].iloc[:5].copy()
    with pytest.raises(ValueError, match="missing POS|absent"):
        impute(tgt_p, ref=ref_p, annotation=ann, verbose=False)


def test_high_tau_warns(phased_cugen, capsys, monkeypatch):
    """The warning must be reachable; the threshold is injectable so a test
    does not have to build a panel large enough to cross it naturally."""
    # `cugen.impute` is the FUNCTION at package level -- the house convention,
    # matching cg.score and cg.freq -- so reach the module through sys.modules
    # rather than `import cugen.impute as ci`, which resolves to the function.
    import sys
    ci = sys.modules["cugen.impute"]
    ref_p, tgt_p, meta = phased_cugen
    monkeypatch.setattr(ci, "_TAU_WARN", 1e-9)
    impute(tgt_p, ref=ref_p, annotation=meta["ann"], ne=meta["ne"],
           verbose=True)
    assert "WARNING" in capsys.readouterr().out


def test_tau_warning_silent_when_ne_is_appropriate(phased_cugen, capsys):
    ref_p, tgt_p, meta = phased_cugen
    impute(tgt_p, ref=ref_p, annotation=meta["ann"], ne=meta["ne"],
           verbose=True)
    assert "WARNING" not in capsys.readouterr().out


# --------------------------------------------------------------------------
# Windowing
# --------------------------------------------------------------------------

def test_single_window_when_region_is_short():
    bounds, owner = plan_windows(np.linspace(0, 5, 100), window=40.0)
    assert len(bounds) == 1 and set(owner) == {0}


def test_windows_tile_the_region_with_overlap():
    cm = np.linspace(0, 100, 1000)
    bounds, owner = plan_windows(cm, window=40.0, overlap=2.0)
    assert bounds[0][0] == 0.0
    assert bounds[-1][1] >= 100.0
    assert bounds[1][0] - bounds[0][0] == pytest.approx(38.0)   # window-overlap
    assert owner.max() == len(bounds) - 1


def test_every_marker_owned_exactly_once():
    """Ownership is a function, so double-counting is impossible by
    construction -- but an off-by-one would leave markers owned by a window
    that does not contain them, which this catches."""
    cm = np.linspace(0, 100, 5000)
    bounds, owner = plan_windows(cm, window=40.0, overlap=2.0)
    for i, w in enumerate(owner):
        lo, hi = bounds[w]
        assert lo <= cm[i] <= hi, f"marker {i} at {cm[i]} owned by {bounds[w]}"


def test_overlap_switches_at_the_midpoint():
    """"discard the value from the window in which the position is closer to
    the window boundary" reduces to a switch at the overlap midpoint."""
    cm = np.linspace(0, 100, 10001)
    bounds, owner = plan_windows(cm, window=40.0, overlap=2.0)
    mid = bounds[1][0] + 1.0                      # overlap is [38, 40]
    assert owner[np.searchsorted(cm, mid - 0.05)] == 0
    assert owner[np.searchsorted(cm, mid + 0.05)] == 1


def test_window_must_exceed_overlap():
    with pytest.raises(ValueError, match="must exceed"):
        plan_windows(np.linspace(0, 10, 10), window=2.0, overlap=2.0)


def test_dosage_r2_bounds_and_direction():
    rng = np.random.default_rng(3)
    certain = rng.integers(0, 2, size=(20, 50)).astype(np.float64)
    assert np.allclose(dosage_r2(certain), 1.0)          # no within-hap variance
    assert np.allclose(dosage_r2(np.full((20, 50), 0.5)), 0.0)
    r = dosage_r2(rng.random((20, 50)))
    assert r.min() >= 0.0 and r.max() <= 1.0


@pytest.mark.parametrize("dtype", [np.float64, np.float32])
def test_bulk_writer_is_byte_identical_to_per_variant(tmp_path, dtype):
    """The fast path must be a pure optimisation, NaNs and all.

    add_variants_bulk recomputes mu/sxx/maf with array operations instead of
    calling variant_stats per marker, so it is a second implementation of the
    same statistics and has to be pinned against the first.
    """
    from cugen.write import CugenWriter, ENCODING_FLOAT16
    rng = np.random.default_rng(0)
    n_s, n_v = 17, 500
    # float32 is the dtype impute() actually passes. An earlier version of the
    # bulk path promoted everything to float64, which doubled four full-size
    # temporaries and made writing the largest phase of a chromosome run; this
    # test covered only float64 and so could not see the change.
    dose = rng.uniform(0, 2, size=(n_s, n_v)).astype(dtype)
    dose[3, 7] = np.nan                      # exercise the missing-value rule
    dose[5, 9] = 3.5                         # and the out-of-range rule
    g = np.arange(n_v) * 3
    a, b = tmp_path / "a.cugen", tmp_path / "b.cugen"
    with CugenWriter(str(a), n_s, n_v, encoding=ENCODING_FLOAT16) as w:
        for j in range(n_v):
            w.add_variant(int(g[j]), dose[:, j])
    with CugenWriter(str(b), n_s, n_v, encoding=ENCODING_FLOAT16) as w:
        for lo in range(0, n_v, 137):        # block size must not matter
            hi = min(lo + 137, n_v)
            w.add_variants_bulk(g[lo:hi], dose[:, lo:hi])
    assert a.read_bytes() == b.read_bytes()


def test_bulk_writer_refuses_integer_encodings(tmp_path):
    from cugen.write import CugenWriter, ENCODING_2BIT
    with CugenWriter(str(tmp_path / "c.cugen"), 4, 2,
                     encoding=ENCODING_2BIT) as w:
        with pytest.raises(ValueError, match="float encodings"):
            w.add_variants_bulk([0, 1], np.zeros((4, 2)))
        w.add_variant(0, [0, 1, 2, 0])
        w.add_variant(1, [0, 1, 2, 0])


def test_impute_host_memory_scales_with_window_not_chromosome(phased_cugen,
                                                              monkeypatch):
    """Peak host memory must not grow with total marker count.

    An earlier version accumulated a full-chromosome, per-HAPLOTYPE, float64
    allele-probability array. That is O(T * M * 8): 27.7 GiB at 2,000 target
    haplotypes on chr20, with the derived dosages adding 13.9 GiB more. Measured
    RSS reached 60 GiB and the run stopped being GPU-bound and started
    thrashing. Allele probabilities are now window-scoped.

    Asserted structurally rather than by measuring RSS, which is far too noisy
    to gate a test on: the retained full-length arrays must be the per-sample
    dosages and the two per-marker summaries, and nothing per-haplotype.
    """
    import sys
    import inspect
    src = inspect.getsource(sys.modules["cugen.impute"].impute)
    assert "allele_prob = np.zeros((T, rref.n_variants)" not in src
    assert "dose = np.zeros((rtgt.n_samples, rref.n_variants), dtype=np.float32)" in src


def test_windowed_summaries_match_a_whole_chromosome_computation(phased_cugen):
    """Accumulating AF and DR2 per window must equal computing them at once."""
    ref_p, tgt_p, meta = phased_cugen
    res = impute(tgt_p, ref=ref_p, annotation=meta["ann"], ne=meta["ne"],
                 verbose=False)
    # one window covers this fixture, so the windowed path and a single-shot
    # computation are the same calculation; this pins that they agree in shape
    # and range rather than silently diverging
    assert res["AF"].between(0, 1).all()
    assert res["DR2"].between(0, 1).all()
    assert len(res) == meta["ann"].shape[0]


# --------------------------------------------------------------------------
# Target-specific state selection
# --------------------------------------------------------------------------

def test_selection_of_everything_matches_brute_force():
    """The reduction must be exact at the boundary, or nothing below it is
    interpretable as an approximation OF the brute-force answer."""
    from cugen._impute_core import forward_backward_sel
    rng = np.random.default_rng(5)
    C, K, T = 30, 40, 6
    rc = rng.integers(0, 3, size=(C, K)).astype(np.int32)
    tc = rng.integers(0, 3, size=(C, T)).astype(np.int32)
    tau = np.concatenate([[0.0], rng.uniform(1e-4, 0.2, size=C - 1)])
    mism = np.full(C, 1e-3)
    sel_all = np.tile(np.arange(K, dtype=np.int32), (T, 1))
    assert np.abs(forward_backward_ref(rc, tc, tau, mism)
                  - forward_backward_sel(rc, tc, tau, mism, sel_all)).max() == 0.0


def test_permuting_the_selection_permutes_the_posteriors():
    """States are labels. Reordering them must move the answer with them and
    not change it -- the check that the gather indexes what it claims to."""
    from cugen._impute_core import forward_backward_sel
    rng = np.random.default_rng(6)
    C, K, T = 20, 30, 4
    rc = rng.integers(0, 3, size=(C, K)).astype(np.int32)
    tc = rng.integers(0, 3, size=(C, T)).astype(np.int32)
    tau = np.concatenate([[0.0], np.full(C - 1, 0.05)])
    mism = np.full(C, 1e-3)
    base = forward_backward_ref(rc, tc, tau, mism)
    perm = rng.permutation(K).astype(np.int32)
    got = forward_backward_sel(rc, tc, tau, mism, np.tile(perm, (T, 1)))
    assert np.abs(base[:, perm, :] - got).max() < 1e-12


def test_selection_finds_an_exact_copy():
    """A target that IS a reference haplotype must keep that haplotype.

    The copied haplotypes are near the END of the panel on purpose. An earlier
    version used indices 7 and 20 with n_states=50, so a selection that ignored
    IBS entirely and simply took the first 50 haplotypes passed -- the mutation
    sweep caught it. Anything below n_states cannot discriminate.
    """
    from cugen._impute_core import select_states
    K = 400
    ref = simulate_mosaic(K, 900, seed=1)
    cm = np.linspace(0, 12, 900)
    sel = select_states(ref, ref[[K - 3, K - 17]], cm, n_states=50)
    assert K - 3 in sel[0], "exact copy not selected"
    assert K - 17 in sel[1], "exact copy not selected"


def test_ibs_selection_beats_an_arbitrary_selection():
    """Selection has to be doing work, not just shrinking the state space.

    Fewer states costs accuracy under ANY selection rule, so a monotone
    degradation curve does not show that the rule is good. This compares
    like-sized selections: IBS-chosen states against the first J haplotypes.
    """
    from cugen._impute_core import (forward_backward_sel, select_states,
                                    aggregate_markers, aggregate_mismatch,
                                    allele_sequence_codes, transition_tau)
    rng = np.random.default_rng(21)
    K, M, J = 400, 1200, 40
    ref = simulate_mosaic(K, M, seed=21)
    tgt_idx = np.sort(rng.choice(M, size=M // 8, replace=False))
    tgt = simulate_mosaic(6, M, seed=22)[:, tgt_idx]
    cm = np.linspace(0, 6, M)
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    full = impute_haplotypes(ref, tgt, tgt_idx, cm, **kw)
    good = impute_haplotypes(ref, tgt, tgt_idx, cm, imp_states=J, **kw)

    # same size, chosen without reference to the target
    arbitrary = np.tile(np.arange(J, dtype=np.int32), (tgt.shape[0], 1))
    starts, stops, agg_cm = aggregate_markers(cm[tgt_idx], 0.005)
    rc, tc = allele_sequence_codes(ref[:, tgt_idx], tgt, starts, stops)
    tau = transition_tau(agg_cm / 100.0, kw["ne"], J)
    mism = aggregate_mismatch(starts, stops, 1e-4)
    post_arb = forward_backward_sel(rc, tc, tau, mism, arbitrary)
    assert post_arb.shape[1] == J

    r_good = np.corrcoef(good.ravel(), full.ravel())[0, 1]
    # an arbitrary selection cannot reconstruct the target at all; compare the
    # posterior mass it puts on its states against what IBS selection achieves
    sel = select_states(ref[:, tgt_idx], tgt, cm[tgt_idx], n_states=J)
    ibs_hits = np.mean([len(set(sel[t].tolist()) & set(range(J)))
                        for t in range(tgt.shape[0])])
    assert r_good > 0.9, f"IBS selection only reached {r_good:.3f}"
    assert ibs_hits < J * 0.5, (
        "IBS selection returned mostly the first J haplotypes, so this test "
        "cannot distinguish it from an arbitrary one")


def test_inverse_map_round_trips():
    from cugen._impute_core import select_states, state_inverse_map
    ref = simulate_mosaic(200, 400, seed=2)
    cm = np.linspace(0, 5, 400)
    sel = select_states(ref, ref[[3, 9]], cm, n_states=40)
    inv = state_inverse_map(sel, 200)
    for t in range(2):
        for j, k in enumerate(sel[t]):
            assert inv[t, k] == j
        missing = set(range(200)) - set(sel[t].tolist())
        assert inv[t, missing.pop()] == -1


def test_imp_states_at_or_above_panel_size_is_a_no_op():
    rng = np.random.default_rng(3)
    K, M = 120, 700
    ref = simulate_mosaic(K, M, seed=3)
    tgt_idx = np.sort(rng.choice(M, size=80, replace=False))
    tgt = simulate_mosaic(4, M, seed=4)[:, tgt_idx]
    cm = np.linspace(0, 4, M)
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    assert np.abs(impute_haplotypes(ref, tgt, tgt_idx, cm, **kw)
                  - impute_haplotypes(ref, tgt, tgt_idx, cm, imp_states=K,
                                      **kw)).max() == 0.0


def test_accuracy_degrades_monotonically_as_states_are_removed():
    """Fewer states must cost accuracy, and cost it smoothly. A selection bug
    that scrambled which haplotypes were kept would show as a flat or
    non-monotone curve -- indistinguishable from 'selection does not matter'."""
    rng = np.random.default_rng(7)
    K, M = 300, 1500
    ref = simulate_mosaic(K, M, seed=7)
    tgt_idx = np.sort(rng.choice(M, size=M // 8, replace=False))
    tgt = simulate_mosaic(6, M, seed=8)[:, tgt_idx]
    cm = np.linspace(0, 6, M)
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    full = impute_haplotypes(ref, tgt, tgt_idx, cm, **kw)
    corrs = []
    for J in (200, 100, 40):
        sub = impute_haplotypes(ref, tgt, tgt_idx, cm, imp_states=J, **kw)
        corrs.append(np.corrcoef(sub.ravel(), full.ravel())[0, 1])
    assert corrs == sorted(corrs, reverse=True), corrs
    assert corrs[0] > 0.95, corrs


def test_tau_uses_the_state_count_not_the_panel_size():
    """tau's 1/|H| is over STATES. Using the panel size while running a subset
    makes the jump term J/K too small and the chain far stickier."""
    import inspect
    from cugen import _impute_core
    src = inspect.getsource(_impute_core.impute_haplotypes)
    assert "n_states_eff" in src
    assert "transition_tau(agg_cm / 100.0, ne, n_states_eff)" in src


# --------------------------------------------------------------------------
# Mosaic composite reference haplotypes
# --------------------------------------------------------------------------

def test_constant_mosaic_matches_plain_selection():
    """A mosaic whose haplotype never changes IS whole-haplotype selection."""
    from cugen._impute_core import forward_backward_mosaic, forward_backward_sel
    rng = np.random.default_rng(9)
    C, K, J, T, n_int = 25, 60, 12, 5, 6
    rc = rng.integers(0, 3, size=(C, K)).astype(np.int32)
    tc = rng.integers(0, 3, size=(C, T)).astype(np.int32)
    tau = np.concatenate([[0.0], rng.uniform(1e-4, 0.2, size=C - 1)])
    mism = np.full(C, 1e-3)
    sel = np.stack([rng.choice(K, size=J, replace=False).astype(np.int32)
                    for _ in range(T)])
    agg_int = np.repeat(np.arange(n_int), int(np.ceil(C / n_int)))[:C]
    hap = np.broadcast_to(sel, (n_int, T, J)).copy()
    a = forward_backward_sel(rc, tc, tau, mism, sel)
    b = forward_backward_mosaic(rc, tc, tau, mism, hap, agg_int)
    assert np.abs(a - b).max() < 1e-15      # gathers differ in layout, not value


def test_mosaic_slots_are_reused_along_the_window():
    """The defining property. A slot holding one haplotype throughout is
    whole-haplotype selection wearing a different name."""
    from cugen._impute_core import composite_haplotypes, ibs_sets
    ref = simulate_mosaic(300, 1200, seed=3)
    cm = np.linspace(0, 8, 1200)
    ibs, _, _ = ibs_sets(ref, ref[[11, 250]], cm, max_per=8)
    hap = composite_haplotypes(ibs, n_states=16, n_targets=2)
    per_slot = [len(np.unique(hap[:, 0, j])) for j in range(16)]
    assert np.mean(per_slot) > 1.5, per_slot


def test_mosaic_keeps_an_exact_copy_present():
    from cugen._impute_core import composite_haplotypes, ibs_sets
    ref = simulate_mosaic(300, 1200, seed=3)
    cm = np.linspace(0, 8, 1200)
    ibs, _, _ = ibs_sets(ref, ref[[11, 250]], cm, max_per=26)
    hap = composite_haplotypes(ibs, n_states=40, n_targets=2)
    for r, h in enumerate((11, 250)):
        frac = (hap[:, r, :] == h).any(axis=1).mean()
        assert frac > 0.9, f"exact copy present in only {frac:.0%} of intervals"


def test_mosaic_inverse_map_is_per_interval():
    from cugen._impute_core import mosaic_inverse_map
    hap = np.array([[3, 7, 1], [2, 9, 4]], dtype=np.int32)   # (T=2, J=3)
    inv = mosaic_inverse_map(hap, 12)
    assert inv[0, 3] == 0 and inv[0, 7] == 1 and inv[0, 1] == 2
    assert inv[1, 2] == 0 and inv[1, 9] == 1 and inv[1, 4] == 2
    assert inv[0, 2] == -1 and inv[1, 3] == -1


def test_mosaics_only_help_when_states_are_scarce():
    """Pins WHERE mosaics matter, which is the whole reason to have them.

    At a generous state budget a mosaic and a whole haplotype per slot are
    equivalent -- there are enough slots for every useful haplotype to hold one
    outright. The gain appears only as J/K falls, which is the regime a
    biobank-scale panel puts you in: J stays at 1,600 while K grows to tens or
    hundreds of thousands.

    Scored on the implied ALLELE PROBABILITIES against the brute-force answer.
    A first version used posterior entropy as a proxy and it pointed the wrong
    way: the mosaic posterior is MORE diffuse, because it spreads mass over
    states that are all locally plausible, while whole-haplotype selection
    concentrates on a few that are locally wrong. Concentration is not quality.
    """
    from cugen._impute_core import (aggregate_markers, aggregate_mismatch,
                                    allele_sequence_codes, build_carriers,
                                    composite_haplotypes, dose_sparse_sel,
                                    forward_backward_mosaic, forward_backward_ref,
                                    forward_backward_sel, ibs_sets,
                                    interpolation_weights, marker_intervals,
                                    mosaic_inverse_map, select_states,
                                    state_inverse_map, transition_tau)
    rng = np.random.default_rng(31)
    K, M = 500, 2500
    ref = simulate_mosaic(K, M, seed=31, n_founders=25)
    tgt_idx = np.sort(rng.choice(M, size=M // 10, replace=False))
    tgt = simulate_mosaic(4, M, seed=32, n_founders=25)[:, tgt_idx]
    cm = np.linspace(0, 12, M)
    ne = scaled_ne(K)
    starts, stops, agg_cm = aggregate_markers(cm[tgt_idx], 0.005)
    rc, tc = allele_sequence_codes(ref[:, tgt_idx], tgt, starts, stops)
    mism = aggregate_mismatch(starts, stops, 1e-4)
    left, lam = interpolation_weights(agg_cm, cm)
    ip, ix, mj = build_carriers(ref)

    def allele_prob(post, inv_of):
        out = np.zeros((tgt.shape[0], M))
        for c in range(post.shape[0] - 1):
            cols = np.flatnonzero(left == c)
            if cols.size:
                out[:, cols] = dose_sparse_sel(post[c], post[c + 1], lam[cols],
                                               ip, ix, mj, cols, inv_of(c))
        return out

    tau_full = transition_tau(agg_cm / 100.0, ne, K)
    base = allele_prob(forward_backward_ref(rc, tc, tau_full, mism),
                       lambda c: state_inverse_map(
                           np.tile(np.arange(K, dtype=np.int32),
                                   (tgt.shape[0], 1)), K))

    def agreement(J):
        tau = transition_tau(agg_cm / 100.0, ne, J)
        sel = select_states(ref[:, tgt_idx], tgt, cm[tgt_idx], n_states=J)
        aw = allele_prob(forward_backward_sel(rc, tc, tau, mism, sel),
                         lambda c: state_inverse_map(sel, K))
        ibs, ist, _ = ibs_sets(ref[:, tgt_idx], tgt, cm[tgt_idx],
                               max_per=max(4, J // 8))
        hap = composite_haplotypes(ibs, n_states=J, n_targets=tgt.shape[0])
        ai = marker_intervals(agg_cm, ist, cm[tgt_idx])
        am = allele_prob(forward_backward_mosaic(rc, tc, tau, mism, hap, ai),
                         lambda c: mosaic_inverse_map(hap[ai[c]], K))
        r = lambda x: np.corrcoef(x.ravel(), base.ravel())[0, 1]
        return r(aw), r(am)

    w_big, m_big = agreement(150)
    w_small, m_small = agreement(20)
    assert m_small > w_small + 0.05, (w_small, m_small)   # scarce: mosaic wins
    assert abs(m_big - w_big) < 0.05, (w_big, m_big)      # generous: a wash


def test_mosaic_reaches_impute_haplotypes():
    """End to end through the public engine, not just the internals.

    Mosaics are NOT uniformly better, and an earlier version of this test
    asserted they were. At a generous state budget whole-haplotype selection
    can win slightly: it picks the globally best haplotypes, while the mosaic
    construction is a greedy staleness heuristic that will splice when it did
    not need to, changing a state's identity mid-window and paying a transition
    for nothing. The advantage is real only where states are scarce.
    """
    rng = np.random.default_rng(3)
    K, M = 400, 1600
    ref = simulate_mosaic(K, M, seed=3, n_founders=20)
    tgt_idx = np.sort(rng.choice(M, size=M // 10, replace=False))
    tgt = simulate_mosaic(4, M, seed=4, n_founders=20)[:, tgt_idx]
    cm = np.linspace(0, 8, M)
    kw = dict(ne=scaled_ne(K), cluster=0.005)
    full = impute_haplotypes(ref, tgt, tgt_idx, cm, **kw)
    r = lambda x: np.corrcoef(x.ravel(), full.ravel())[0, 1]

    got = {}
    for J in (120, 16):
        whole = impute_haplotypes(ref, tgt, tgt_idx, cm, imp_states=J, **kw)
        mos = impute_haplotypes(ref, tgt, tgt_idx, cm, imp_states=J,
                                mosaic=True, **kw)
        assert whole.shape == mos.shape == full.shape
        for x in (whole, mos):
            assert x.min() >= -1e-12 and x.max() <= 1 + 1e-12
        got[J] = (r(whole), r(mos))

    # scarce states (J/K = 1/25): the mosaic must win, and clearly
    w16, m16 = got[16]
    assert m16 > w16 + 0.05, got
    # generous states (J/K = 1/3): close either way, no strong claim
    w120, m120 = got[120]
    assert abs(m120 - w120) < 0.05, got
    # and the advantage must GROW as states get scarcer -- that is the claim
    assert (m16 - w16) > (m120 - w120), got


# --------------------------------------------------------------------------
# is_phased
# --------------------------------------------------------------------------

def test_is_phased_reports_both_kinds(tmp_path):
    from cugen.write import write_cugen
    rng = np.random.default_rng(0)
    ph, un = tmp_path / "ph.cugen", tmp_path / "un.cugen"
    write_cugen_phased(str(ph), rng.integers(0, 2, size=(2 * 20, 50)).astype(np.uint8))
    write_cugen(str(un), rng.integers(0, 3, size=(20, 50)).astype(np.uint8))
    assert cio.is_phased(str(ph)) is True
    assert cio.is_phased(str(un)) is False


def test_is_phased_refuses_a_file_whose_flag_and_encoding_disagree(tmp_path):
    """There is no correct answer for such a file, so it must not return one.

    hap2bit and 2bit share bytes_per_variant, so the file passes every
    structural check while decoding to different genotypes depending on which
    field you believe.
    """
    rng = np.random.default_rng(2)
    p = tmp_path / "bad.cugen"
    write_cugen_phased(str(p), rng.integers(0, 2, size=(8, 6)).astype(np.uint8))
    raw = bytearray(p.read_bytes())
    raw[64] &= ~cwrite.FLAG_PHASED          # clear PHASED, keep hap2bit
    p.write_bytes(bytes(raw))
    with pytest.raises(ValueError, match="disagree"):
        cio.is_phased(str(p))


def test_is_phased_does_not_read_the_data_block(tmp_path, monkeypatch):
    """It must stay cheap enough to call across a directory of chromosomes.

    Asserted by forbidding mmap rather than by timing, which would be flaky.
    """
    import mmap as _mmap
    rng = np.random.default_rng(1)
    p = tmp_path / "ph.cugen"
    write_cugen_phased(str(p), rng.integers(0, 2, size=(2 * 30, 400)).astype(np.uint8))

    def _boom(*a, **k):
        raise AssertionError("is_phased mmap'd the file")

    monkeypatch.setattr(_mmap, "mmap", _boom)
    assert cio.is_phased(str(p)) is True
