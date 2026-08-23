"""VCF -> .cugen dosage conversion.

The genotype codes are the whole risk here. cyvcf2's `gt_types` is
0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT by default, and only becomes the
intuitive 0/1/2-plus-3-for-missing when the VCF is opened with gts012=True.
Getting that backwards maps every homozygous-ALT call to missing and every
missing call to homozygous-ALT -- silently, with plausible-looking output.
"""
import numpy as np
import pytest

cyvcf2 = pytest.importorskip("cyvcf2")

from cugen.convert import vcf2cugen          # noqa: E402
from cugen.io import read_cugen              # noqa: E402
from cugen.write import unpack_2bit          # noqa: E402

# One variant per genotype class, plus a mixed one, and a missing call.
VCF = """##fileformat=VCFv4.2
##contig=<ID=chr22>
##FORMAT=<ID=GT,Number=1,Type=String,Description="Genotype">
#CHROM\tPOS\tID\tREF\tALT\tQUAL\tFILTER\tINFO\tFORMAT\ts1\ts2\ts3\ts4
chr22\t100\tv0\tA\tG\t.\t.\t.\tGT\t0|0\t0|1\t1|0\t1|1
chr22\t200\tv1\tC\tT\t.\t.\t.\tGT\t1|1\t1|1\t1|1\t1|1
chr22\t300\tv2\tG\tA\t.\t.\t.\tGT\t0/0\t0/1\t1/1\t./.
"""

# what each row must decode to: dosage 0/1/2, 3 = missing
WANT = np.array([[0, 1, 1, 2],
                 [2, 2, 2, 2],
                 [0, 1, 2, 3]], dtype=np.uint8)


def _decode(path, n, p):
    r = read_cugen(path)
    packed = np.frombuffer(r.read_packed_bytes(), dtype=np.uint8)
    bpv = int(r.bytes_per_variant)
    return np.stack([unpack_2bit(packed[v * bpv:(v + 1) * bpv], n)
                     for v in range(p)])


def test_homozygous_alt_is_dosage_two_not_missing(tmp_path):
    """The bug this file exists for.

    Measured on real 1000 Genomes chr22 before the fix: a variant with 2,321
    `1|1` genotypes produced 2,321 missing calls and ZERO dosage-2 calls.
    """
    src = tmp_path / "t.vcf"
    src.write_text(VCF)
    out = str(tmp_path / "t.cugen")
    vcf2cugen(str(src), out, verbose=False)

    got = _decode(out, 4, 3)
    np.testing.assert_array_equal(got, WANT)


def test_the_all_homozygous_alt_variant_is_not_reported_as_missing(tmp_path):
    """Sharpest form: a variant where every call is 1|1 must have allele
    frequency 1.0 and no missingness -- not 0.0 and entirely missing."""
    src = tmp_path / "t.vcf"
    src.write_text(VCF)
    out = str(tmp_path / "t.cugen")
    vcf2cugen(str(src), out, verbose=False)

    got = _decode(out, 4, 3)
    assert (got[1] == 2).all(), f"all-1|1 variant decoded as {got[1]}"
    assert (got[1] == 3).sum() == 0


def test_a_vcf_with_no_missing_calls_does_not_set_has_missing(tmp_path):
    """Consequence worth pinning separately: a spurious HAS_MISSING flag
    silently disables the fused GPU path, and with it stream= and count_only.
    """
    src = tmp_path / "clean.vcf"
    src.write_text("\n".join(VCF.splitlines()[:-1]) + "\n")   # drop the ./. row
    out = str(tmp_path / "clean.cugen")
    vcf2cugen(str(src), out, verbose=False)
    assert not read_cugen(out).has_missing, (
        "HAS_MISSING set on a VCF with no missing calls; this makes the fused "
        "scan, stream=True and count_only unreachable")
