"""cugen.genotypes block readers and the shared 2-bit codec (CPU)."""
import numpy as np
import pandas as pd
import pytest

from conftest import a2_counts, write_bed
from cugen import _codec
from cugen.genotypes import BedGenotypes, CugenGenotypes, open_genotypes


# ----------------------------------------------------------------------------- codec
def test_codec_roundtrips():
    rng = np.random.default_rng(0)
    for n in (1, 3, 4, 5, 37):
        codes = rng.integers(0, 4, size=(6, n)).astype(np.uint8)
        np.testing.assert_array_equal(_codec.unpack_cugen(_codec.pack_cugen(codes), n), codes)
        for counted in ("A1", "A2"):
            np.testing.assert_array_equal(
                _codec.unpack_bed(_codec.pack_bed(codes, counted), n, counted), codes)
    with pytest.raises(ValueError):
        _codec.bed_lut("ALT")


def test_codec_layouts_match_the_formats():
    # cugen: sample 0 in the high bits; bed: sample 0 in the low bits, 01 = missing
    assert _codec.pack_cugen(np.array([[1, 2, 3, 0]]))[0, 0] == 0b01101100
    assert list(_codec.unpack_bed(np.array([[0b11100100]], np.uint8), 4, "A1")[0]) == [2, 3, 1, 0]
    assert list(_codec.unpack_bed(np.array([[0b11100100]], np.uint8), 4, "A2")[0]) == [0, 3, 1, 2]
    # hap2bit: popcount, no missing code
    assert list(_codec.unpack_hap2bit(np.array([[0b00011011]], np.uint8), 4)[0]) == [0, 1, 1, 2]


def test_write_pack_helpers_unchanged():
    from cugen.write import pack_2bit, unpack_2bit
    g = np.array([0, 1, 2, 3, 2, 1], dtype=np.uint8)
    r = np.concatenate([g, np.zeros(2, np.uint8)]).reshape(-1, 4)
    ref = ((r[:, 0] << 6) | (r[:, 1] << 4) | (r[:, 2] << 2) | r[:, 3]).astype(np.uint8)
    np.testing.assert_array_equal(pack_2bit(g), ref)
    np.testing.assert_array_equal(unpack_2bit(ref.tobytes(), 6), g)


# ----------------------------------------------------------------------------- readers
def test_decode_bed_cugen_and_sample_gather(cohort):
    G = cohort["G"]
    np.testing.assert_array_equal(cohort["bed"].read_counts(), G)
    np.testing.assert_array_equal(cohort["cg"].read_counts(), a2_counts(G))
    ids = [f"s{i}" for i in (5, 2, 9, 0)]
    sub = BedGenotypes(cohort["dir"] + "/t", samples=ids)
    np.testing.assert_array_equal(sub.read_counts(), G[:, [5, 2, 9, 0]])
    assert not sub.sample_order_is_file_order
    a2 = BedGenotypes(cohort["dir"] + "/t", counted="A2")
    np.testing.assert_array_equal(a2.read_counts(), a2_counts(G))
    assert cohort["bed"].counted == "A1" and "column 5" in cohort["bed"].counted_label
    assert cohort["cg"].counted == "ALT"


def test_scattered_reads_match_full_decode(cohort):
    rows = np.array([0, 1, 5, 69, 70, 71, 140, 169])           # gaps, file boundaries (cugen)
    for g, truth in ((cohort["bed"], cohort["G"]), (cohort["cg"], a2_counts(cohort["G"]))):
        np.testing.assert_array_equal(g.read_counts(rows), truth[rows])
    assert cohort["bed"].resident_bytes() > 0


def test_snps_selection_by_id_file_and_chromosomes(cohort, tmp_path):
    f = tmp_path / "x.prune.in"
    f.write_text("snp3\nsnp10\nsnp150\nnot_there\n")
    bed = cohort["bed"]
    np.testing.assert_array_equal(bed.select(str(f)), [3, 10, 150])
    np.testing.assert_array_equal(bed.select(["snp7"], chromosomes=[1]), [7])
    assert bed.select(chromosomes=[2]).tolist() == list(range(70, 125))
    mask = np.zeros(bed.n_variants, bool)
    mask[[1, 2]] = True
    np.testing.assert_array_equal(bed.select(mask), [1, 2])
    with pytest.raises(ValueError):
        bed.select(["nope"])


def test_cugen_chromosomes_subset(cohort):
    cg = CugenGenotypes(cohort["dir"], chromosomes=[1, 3])
    assert sorted(np.unique(cg.chrom)) == [1, 3]
    o = np.isin(cohort["chrom"], [1, 3])
    np.testing.assert_array_equal(cg.read_counts(), a2_counts(cohort["G"][o]))


def test_open_genotypes_dispatch(cohort):
    assert isinstance(open_genotypes(cohort["dir"] + "/t"), BedGenotypes)
    assert isinstance(open_genotypes(cohort["dir"] + "/t.bed"), BedGenotypes)
    assert isinstance(open_genotypes(cohort["dir"]), CugenGenotypes)
    with pytest.raises(FileNotFoundError):
        open_genotypes(cohort["dir"] + "/missing")


def test_cugen_annotation_ids_and_alleles(cohort, tmp_path):
    n_var = len(cohort["chrom"])
    ann = pd.DataFrame({"gidx": np.arange(n_var), "CHR": cohort["chrom"], "POS": np.arange(n_var) * 10 + 5,
                        "ID": [f"v{i}" for i in range(n_var)], "REF": ["A"] * n_var, "ALT": ["G"] * n_var})
    cg = CugenGenotypes(cohort["dir"], annotation=ann)
    np.testing.assert_array_equal(cg.select(["v3", "v100"]), [3, 100])
    cnt, oth = cg.alleles([0])
    assert cnt.to_pylist() == ["G"] and oth.to_pylist() == ["A"]
    assert cg.pos[7] == 75


def test_phased_cugen_reads_popcount_dosages(tmp_path, phased_panel):
    from cugen.write import write_cugen_phased
    ref_hap = phased_panel[0][:40, :64]                    # 20 samples x 64 markers
    p = tmp_path / "chr20.cugen"
    write_cugen_phased(str(p), ref_hap)
    g = CugenGenotypes([str(p)])
    assert g.phased and g.encoding == "hap2bit"
    np.testing.assert_array_equal(g.read_counts(), (ref_hap[0::2] + ref_hap[1::2]).T)
    mu, sxx, maf = g.stored_stats(np.arange(4))
    np.testing.assert_allclose(mu, (ref_hap[0::2] + ref_hap[1::2]).mean(0)[:4], rtol=1e-6)


def test_bed2cugen_counted_vocabulary(cohort, tmp_path):
    """bed2cugen(counted=) and BedGenotypes(counted=) agree; the annotation names the counted allele."""
    from cugen import bed2cugen
    o = cohort["chrom"] == 1
    pre = str(tmp_path / "c1")
    write_bed(pre, cohort["G"][o], cohort["chrom"][o], a1=["C"] * int(o.sum()), a2=["T"] * int(o.sum()))
    for counted in ("A1", "A2"):
        out = str(tmp_path / counted / "chr1.cugen")
        import os
        os.makedirs(os.path.dirname(out))
        bed2cugen(pre + ".bed", out, verbose=False, counted=counted, annotation=out + ".feather")
        cg = CugenGenotypes([out])
        np.testing.assert_array_equal(cg.read_counts(), BedGenotypes(pre, counted=counted).read_counts())
        ann = pd.read_feather(out + ".feather")
        assert ann["ALT"].iloc[0] == ("C" if counted == "A1" else "T")
        assert ann["REF"].iloc[0] == ("T" if counted == "A1" else "C")
        assert ann["gidx"].tolist() == list(range(int(o.sum())))
