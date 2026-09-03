"""Merging per-chromosome .cugen files into one genome-wide file.

Cross-chromosome LD is not expressible against a directory of files --
ld_matrix takes a single .cugen -- so a genome-wide all-pairs scan needs the
variants in one file, with gidx numbered continuously across chromosomes.
"""
import numpy as np
import pytest

from cugen import ld as L
from cugen.write import write_cugen, write_cugen_phased
from conftest import simulate_haplotypes

CPU = dict(backend="numpy", verbose=False)


def test_merge_concatenates_variants_and_renumbers_gidx(tmp_path):
    """The merged file holds every variant, in order, with continuous gidx."""
    from cugen.convert import merge_cugen

    a = simulate_haplotypes(30, 7, seed=1)
    b = simulate_haplotypes(30, 5, seed=2)
    pa, pb = tmp_path / "a.cugen", tmp_path / "b.cugen"
    write_cugen(str(pa), a.T)
    write_cugen(str(pb), b.T)

    out = tmp_path / "genome.cugen"
    merge_cugen([str(pa), str(pb)], str(out), verbose=False)

    import cugen as cg
    h = cg.io.read_cugen_header(str(out))
    assert int(h["n_variants"]) == 12
    assert int(h["n_samples"]) == 30
    r = cg.io.read_cugen(str(out))
    assert list(np.asarray(r.gidx)) == list(range(12)), "gidx must be continuous"


def test_merged_dosages_match_the_sources(tmp_path):
    """Every decoded dosage survives the merge unchanged."""
    from cugen.convert import merge_cugen

    a = simulate_haplotypes(24, 6, seed=3)
    b = simulate_haplotypes(24, 4, seed=4)
    pa, pb = tmp_path / "a.cugen", tmp_path / "b.cugen"
    write_cugen(str(pa), a.T)
    write_cugen(str(pb), b.T)
    out = tmp_path / "g.cugen"
    merge_cugen([str(pa), str(pb)], str(out), verbose=False)

    import cugen as cg
    got = L._dosages_numpy(cg.io.read_cugen(str(out)))
    want = np.concatenate([a, b], axis=0)          # (n_variants, n_samples)
    np.testing.assert_array_equal(got[:, :24], want[:, :24])


def test_merge_refuses_mismatched_sample_counts(tmp_path):
    """Two cohorts cannot be concatenated on the variant axis."""
    from cugen.convert import merge_cugen

    pa, pb = tmp_path / "a.cugen", tmp_path / "b.cugen"
    write_cugen(str(pa), simulate_haplotypes(20, 4, seed=5).T)
    write_cugen(str(pb), simulate_haplotypes(21, 4, seed=6).T)
    with pytest.raises(ValueError, match="n_samples"):
        merge_cugen([str(pa), str(pb)], str(tmp_path / "g.cugen"), verbose=False)


def test_merge_refuses_mixed_encodings(tmp_path):
    """A phased and an unphased file share bytes but not meaning."""
    from cugen.convert import merge_cugen

    pa = tmp_path / "unphased.cugen"
    pb = tmp_path / "phased.cugen"
    write_cugen(str(pa), simulate_haplotypes(16, 4, seed=7).T)
    hap = (np.random.default_rng(8).random((32, 4)) < 0.5).astype(np.uint8)
    write_cugen_phased(str(pb), hap)
    with pytest.raises(ValueError, match="encoding"):
        merge_cugen([str(pa), str(pb)], str(tmp_path / "g.cugen"), verbose=False)


def test_cross_chromosome_ld_matches_within_file_ld(tmp_path):
    """LD over the merged file reproduces LD computed on one source file.

    This is the property that makes a genome-wide scan meaningful: pairs that
    lived inside one chromosome must be unchanged by the merge, and pairs that
    span the join must now exist.
    """
    from cugen.convert import merge_cugen

    a = simulate_haplotypes(40, 6, seed=11)
    b = simulate_haplotypes(40, 6, seed=12)
    pa, pb = tmp_path / "a.cugen", tmp_path / "b.cugen"
    write_cugen(str(pa), a.T)
    write_cugen(str(pb), b.T)
    out = tmp_path / "g.cugen"
    merge_cugen([str(pa), str(pb)], str(out), verbose=False)

    solo = L.ld_matrix(str(pa), stats=("r", "r2"), min_r2=0.0, min_obs=1, **CPU)
    both = L.ld_matrix(str(out), stats=("r", "r2"), min_r2=0.0, min_obs=1, **CPU)

    solo_r = {(int(x), int(y)): float(v) for x, y, v in
              zip(solo["gidx_a"], solo["gidx_b"], solo["R"])}
    both_r = {(int(x), int(y)): float(v) for x, y, v in
              zip(both["gidx_a"], both["gidx_b"], both["R"])}
    for k, v in solo_r.items():
        assert k in both_r, f"within-chromosome pair {k} lost in the merge"
        assert both_r[k] == pytest.approx(v, rel=1e-6), f"pair {k} changed"
    # and the join is now spanned
    assert any(x < 6 <= y for x, y in both_r), "no cross-file pairs were produced"
