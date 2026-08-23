"""cugen.convert - build .cugen files from PGEN, VCF/BCF or PLINK1 BED.

    pgen2cugen   PLINK2 .pgen/.pvar/.psam   (needs pgenlib)
    vcf2cugen    VCF/VCF.gz/BCF             (needs cyvcf2, else pysam)
    bed2cugen    PLINK1 .bed/.bim/.fam      (no dependencies -- pure numpy)

All three stream variant-by-variant through CugenWriter, so peak memory is one
variant, not the genotype matrix.

SAMPLE ORDER IS THE CONTRACT
----------------------------
Every downstream cugen object -- cohort .npz `sample_idx_sorted`, the LOCO
matrices, the covariate sidecar -- is positional against the sample order of
the .cugen it was built from. Converting the same source with a different
--keep list produces a file that is structurally valid and silently
incompatible with those objects. Each converter writes a companion
`<out>.samples.txt` recording the exact order used; check it before pairing a
new .cugen with existing cohorts.

MISSING GENOTYPES
-----------------
Encoded as 2-bit value 3 and EXCLUDED from mu_x/sxx/maf. The association
kernels are complete-case (io.py:108): they skip 3 rather than treating it as
dosage 0. Writing missing as 0 instead would inflate the effective variance
relative to the stored non-missing sxx and manufacture false positives on
high-missingness probes -- that failure mode cost a lot of debugging, so do not
"simplify" it here.

BED note: PLINK1 is already 2-bit but uses a DIFFERENT code (01 = missing,
little-endian within byte). bed2cugen remaps rather than copying bytes.
"""
import os
import sys

import numpy as np

from .write import CugenWriter, ENCODING_2BIT


def _write_samples(out_path, sample_ids):
    p = f"{out_path}.samples.txt"
    with open(p, "w") as f:
        f.write("\n".join(str(s) for s in sample_ids) + "\n")
    return p


def _progress(i, n, every=50000):
    if n and i % every == 0:
        print(f"  {i:,}/{n:,} variants ({100.0*i/n:.0f}%)", flush=True)


def pgen2cugen(pgen, out, psam=None, pvar=None, sample_idx=None,
               variant_idx=None, gidx=None, verbose=True):
    """PLINK2 .pgen -> .cugen. Requires pgenlib."""
    try:
        import pgenlib
    except ImportError:
        raise ImportError("pgen2cugen needs pgenlib (pip install Pgenlib)")

    psam = psam or os.path.splitext(pgen)[0] + ".psam"
    ids = []
    with open(psam) as f:
        for line in f:
            if line.startswith("##"):
                continue
            p = line.split()
            if p[0].lstrip("#").upper() in ("FID", "IID"):
                continue
            ids.append(p[0])
    raw_n = len(ids)

    si = None if sample_idx is None else np.asarray(sample_idx, dtype=np.uint32)
    reader = pgenlib.PgenReader(str(pgen).encode(), raw_sample_ct=raw_n,
                                sample_subset=si)
    n_samples = reader.get_raw_sample_ct() if si is None else len(si)
    total = reader.get_variant_ct()
    vidx = np.arange(total) if variant_idx is None else np.asarray(variant_idx)
    g = vidx if gidx is None else np.asarray(gidx)
    kept_ids = ids if si is None else [ids[i] for i in si]

    if verbose:
        print(f"pgen2cugen: {n_samples:,} samples x {len(vidx):,} variants -> {out}")
    buf = np.empty(n_samples, dtype=np.int8)
    with CugenWriter(out, n_samples, len(vidx), ENCODING_2BIT) as w:
        for k, v in enumerate(vidx):
            reader.read(int(v), buf)
            d = buf.astype(np.float64)
            d[d < 0] = 3.0                      # pgenlib marks missing as -9
            w.add_variant(int(g[k]), d)
            if verbose:
                _progress(k, len(vidx))
    reader.close()
    _write_samples(out, kept_ids)
    return out


def vcf2cugen(vcf, out, region=None, keep=None, gidx_start=0,
              min_ac=0, verbose=True):
    """VCF/BCF -> .cugen. Uses cyvcf2 if available, else pysam.

    Two passes: one to count variants (the header does not carry the count),
    one to write. Pass --region to convert a chromosome at a time, which is how
    the rest of cugen expects the data laid out (chr<N>.cugen).
    """
    try:
        from cyvcf2 import VCF
        backend = "cyvcf2"
    except ImportError:
        try:
            import pysam
            backend = "pysam"
        except ImportError:
            raise ImportError("vcf2cugen needs cyvcf2 or pysam")

    def _open():
        # gts012=True is load-bearing, not a preference. cyvcf2's DEFAULT
        # gt_types encoding is 0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT; only
        # with gts012=True does it become 0/1/2 dosage plus 3 for missing,
        # which is what the mapping below assumes and what ENCODING_2BIT
        # means. Without it every homozygous-ALT call is written as MISSING and
        # every missing call as dosage 2 -- silently, with plausible output.
        # Measured on real 1000 Genomes chr22: a variant with 2,321 `1|1`
        # genotypes produced 2,321 missing and zero dosage-2, and the spurious
        # HAS_MISSING flag then disabled the fused GPU scan, stream= and
        # count_only for the whole file.
        #
        # vcf2cugenh does not need this: it reads allele indices through
        # rec.genotypes, which gts012 does not affect.
        return (VCF(vcf, gts012=True) if backend == "cyvcf2"
                else pysam.VariantFile(vcf))

    v = _open()
    samples = list(v.samples if backend == "cyvcf2" else v.header.samples)
    keep_idx = None
    if keep:
        want = set(keep if isinstance(keep, (list, set, tuple))
                   else open(keep).read().split())
        keep_idx = [i for i, s in enumerate(samples) if s in want]
        samples = [samples[i] for i in keep_idx]
    n_samples = len(samples)

    it = (v(region) if region else v) if backend == "cyvcf2" else \
        (v.fetch(region=region) if region else v)
    n_var = sum(1 for _ in it)
    v.close() if backend == "cyvcf2" else v.close()
    if verbose:
        print(f"vcf2cugen [{backend}]: {n_samples:,} samples x {n_var:,} "
              f"variants{f' in {region}' if region else ''} -> {out}")

    v = _open()
    it = (v(region) if region else v) if backend == "cyvcf2" else \
        (v.fetch(region=region) if region else v)
    with CugenWriter(out, n_samples, n_var, ENCODING_2BIT) as w:
        for k, rec in enumerate(it):
            if backend == "cyvcf2":
                # gts012=True, so 0,1,2 are dosages and 3 is UNKNOWN.
                gt = rec.gt_types.astype(np.float64)
                d = np.where(gt == 3, 3.0, gt)
            else:
                d = np.empty(len(rec.samples), dtype=np.float64)
                for i, s in enumerate(rec.samples.values()):
                    a = s.get("GT")
                    d[i] = 3.0 if a is None or None in a else float(sum(a))
            if keep_idx is not None:
                d = d[keep_idx]
            w.add_variant(gidx_start + k, d)
            if verbose:
                _progress(k, n_var)
    v.close()
    _write_samples(out, samples)
    return out


def bed2cugen(bed, out, bim=None, fam=None, gidx_start=0, verbose=True):
    """PLINK1 .bed -> .cugen. Pure numpy, no external dependency.

    PLINK1 packs 2 bits per genotype LITTLE-endian within the byte, with codes
    00=hom-A1, 01=MISSING, 10=het, 11=hom-A2. cugen uses BIG-endian within the
    byte with 0/1/2 dosage and 3=missing, so every genotype is remapped.
    """
    bim = bim or os.path.splitext(bed)[0] + ".bim"
    fam = fam or os.path.splitext(bed)[0] + ".fam"
    ids = [l.split()[1] for l in open(fam) if l.strip()]
    n_samples = len(ids)
    n_var = sum(1 for l in open(bim) if l.strip())

    with open(bed, "rb") as f:
        magic = f.read(3)
        if magic[:2] != b"\x6c\x1b":
            raise ValueError(f"{bed}: not a PLINK1 .bed (magic {magic[:2]!r})")
        if magic[2] != 1:
            raise ValueError(f"{bed}: sample-major .bed unsupported; "
                             f"re-encode with --make-bed")
        bpv = (n_samples + 3) // 4
        if verbose:
            print(f"bed2cugen: {n_samples:,} samples x {n_var:,} variants -> {out}")
        # PLINK1 code -> dosage; 1 (01) is missing
        lut = np.array([0.0, 3.0, 1.0, 2.0], dtype=np.float64)
        with CugenWriter(out, n_samples, n_var, ENCODING_2BIT) as w:
            for k in range(n_var):
                raw = np.frombuffer(f.read(bpv), dtype=np.uint8)
                codes = np.empty(raw.size * 4, dtype=np.uint8)
                codes[0::4] = raw & 3           # little-endian within byte
                codes[1::4] = (raw >> 2) & 3
                codes[2::4] = (raw >> 4) & 3
                codes[3::4] = (raw >> 6) & 3
                w.add_variant(gidx_start + k, lut[codes[:n_samples]])
                if verbose:
                    _progress(k, n_var)
    _write_samples(out, ids)
    return out


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(prog="cugen-convert",
                                 description="build .cugen from pgen/vcf/bed")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pgen"); p.add_argument("input"); p.add_argument("output")
    p.add_argument("--psam"); p.add_argument("--pvar")
    v = sub.add_parser("vcf"); v.add_argument("input"); v.add_argument("output")
    v.add_argument("--region"); v.add_argument("--keep")
    v.add_argument("--gidx-start", type=int, default=0)
    b = sub.add_parser("bed"); b.add_argument("input"); b.add_argument("output")
    b.add_argument("--bim"); b.add_argument("--fam")
    b.add_argument("--gidx-start", type=int, default=0)
    a = ap.parse_args(argv)

    if a.cmd == "pgen":
        pgen2cugen(a.input, a.output, psam=a.psam, pvar=a.pvar)
    elif a.cmd == "vcf":
        vcf2cugen(a.input, a.output, region=a.region, keep=a.keep,
                  gidx_start=a.gidx_start)
    else:
        bed2cugen(a.input, a.output, bim=a.bim, fam=a.fam,
                  gidx_start=a.gidx_start)
    from .write import validate_cugen
    validate_cugen(a.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())


def vcf2cugenh(vcf, out, region=None, keep=None, gidx_start=0,
               gidx=None, require_phased=True, verbose=True):
    """PHASED VCF/BCF -> phased .cugen (ENCODING_HAP2BIT).

    The counterpart of vcf2cugen for haplotype data. Every genotype must be
    diploid, phased and non-missing -- the requirement Beagle places on a
    reference panel, and the one cugen.impute inherits.

    require_phased=True rejects a record carrying an unphased genotype. That
    check matters more than it looks: an unphased het written as if it were
    phased silently invents a haplotype assignment, and the resulting panel
    still imputes, still returns probabilities in [0, 1], and is simply wrong in
    a way no downstream check can detect. Stock 1000 Genomes VCFs are phased;
    many other sources are not, and some are phased for homozygotes only.

    Multi-allelic records are skipped: hap2bit stores one bit per haplotype and
    so is biallelic by construction. Split them first with
    `bcftools norm -m -any` if you need them.

    `gidx` gives an explicit global index per written record, which matters
    whenever this file will be matched against another: cugen.impute pairs a
    target with its reference THROUGH gidx, so numbering each file 0..n-1
    independently silently aligns target marker 0 with reference marker 0 and
    shifts every marker after it. Nothing fails; the imputation simply uses the
    wrong reference markers.
    """
    try:
        from cyvcf2 import VCF
        backend = "cyvcf2"
    except ImportError:
        try:
            import pysam
            backend = "pysam"
        except ImportError:
            raise ImportError("vcf2cugenh needs cyvcf2 or pysam")

    from .write import ENCODING_HAP2BIT

    def _open():
        return VCF(vcf) if backend == "cyvcf2" else pysam.VariantFile(vcf)

    def _iter(v):
        if backend == "cyvcf2":
            return v(region) if region else v
        return v.fetch(region=region) if region else v

    def _biallelic(rec):
        alts = rec.ALT if backend == "cyvcf2" else list(rec.alts or ())
        return len(alts) == 1

    v = _open()
    samples = list(v.samples if backend == "cyvcf2" else v.header.samples)
    keep_idx = None
    if keep:
        want = set(keep if isinstance(keep, (list, set, tuple))
                   else open(keep).read().split())
        keep_idx = [i for i, s in enumerate(samples) if s in want]
        samples = [samples[i] for i in keep_idx]
    n_samples = len(samples)

    n_var = n_skip = 0
    for rec in _iter(v):
        if _biallelic(rec):
            n_var += 1
        else:
            n_skip += 1
    v.close()
    if verbose:
        print(f"vcf2cugenh [{backend}]: {n_samples:,} samples x {n_var:,} "
              f"biallelic variants{f' in {region}' if region else ''} -> {out}"
              + (f"  ({n_skip:,} multi-allelic skipped)" if n_skip else ""))
    if n_var == 0:
        raise ValueError(f"{vcf}: no biallelic records"
                         f"{f' in {region}' if region else ''}")

    if gidx is not None:
        gidx = np.asarray(gidx, dtype=np.int64)
        if gidx.size != n_var:
            raise ValueError(
                f"gidx has {gidx.size} entries but {n_var} biallelic records "
                f"will be written")
    v = _open()
    k = 0
    with CugenWriter(out, n_samples, n_var, ENCODING_HAP2BIT) as w:
        for rec in _iter(v):
            if not _biallelic(rec):
                continue
            if backend == "cyvcf2":
                # rec.genotype.array() hands back a numpy array directly;
                # rec.genotypes builds a Python list-of-lists per record, and
                # np.asarray on it dominates the whole conversion at
                # chromosome scale. Same layout either way: allele columns
                # then a phase flag, with -1 for missing.
                ga = getattr(rec, "genotype", None)
                if ga is not None and hasattr(ga, "array"):
                    arr = ga.array()
                else:
                    arr = np.asarray(rec.genotypes, dtype=np.int16)
                a = arr[:, :2].astype(np.int64, copy=False)
                phased = arr[:, -1].astype(bool)
            else:
                a = np.empty((len(rec.samples), 2), dtype=np.int64)
                phased = np.empty(len(rec.samples), dtype=bool)
                for i, s in enumerate(rec.samples.values()):
                    al = s.get("GT")
                    a[i] = (-1, -1) if al is None or len(al) != 2 else al
                    phased[i] = bool(s.phased)
            if keep_idx is not None:
                a = a[keep_idx]
                phased = phased[keep_idx]
            if (a < 0).any():
                bad = int(np.flatnonzero((a < 0).any(axis=1))[0])
                raise ValueError(
                    f"{vcf}: missing genotype for sample {samples[bad]!r} at "
                    f"record {k} ({rec.CHROM if backend == 'cyvcf2' else rec.chrom}"
                    f":{rec.POS if backend == 'cyvcf2' else rec.pos}). A phased "
                    f".cugen has no code for missing; impute or drop first.")
            # Homozygotes are unambiguous whether or not the record marks them
            # phased, so only heterozygotes need the flag.
            if require_phased:
                het = a[:, 0] != a[:, 1]
                if np.any(het & ~phased):
                    bad = int(np.flatnonzero(het & ~phased)[0])
                    raise ValueError(
                        f"{vcf}: UNPHASED heterozygote for sample "
                        f"{samples[bad]!r} at record {k}. Writing it as phased "
                        f"would invent a haplotype assignment that was never "
                        f"observed, and the panel would still impute and still "
                        f"look correct. Phase first, or pass "
                        f"require_phased=False if you know what you are doing.")
            g = gidx_start + k if gidx is None else int(gidx[k])
            w.add_variant_phased(g, a.astype(np.uint8))
            k += 1
            if verbose:
                _progress(k, n_var)
    v.close()
    _write_samples(out, samples)
    return out
