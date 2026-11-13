"""cugen.write - create .cugen files.

The public package could read .cugen but not write one: ``io.write_cugen``
raised NotImplementedError and pointed at legacy ``build_*_cugen.sh`` scripts
that are not in the repository. So anyone outside this cluster could consume
the format but never produce it. This module closes that.

    CugenWriter        streaming writer, one variant at a time
    write_cugen        array -> file convenience wrapper
    validate_cugen     structural check of an existing file

ON-DISK FORMAT (v1) - little-endian throughout
----------------------------------------------
    offset  size  type      field
         0     8  char[8]   magic "CUPGEN01"   (legacy magic, kept post-rename)
         8     4  uint32    version = 1
        12     4  uint32    encoding            0=2bit 1=uint8 2=f16 3=f32
                                                4=hap2bit (phased haplotypes)
        16     8  uint64    n_samples
        24     8  uint64    n_variants
        32     8  uint64    bytes_per_variant
        40     8  uint64    stats_offset
        48     8  uint64    data_offset
        56     8  uint64    gidx_offset
        64     4  uint32    flags               bit0 HAS_MISSING, bit1 HAS_GIDX,
                                                bit2 PHASED
        68   188  --        zero padding to 256

    stats  [stats_offset]  float32 x n_variants x 3, as three CONTIGUOUS arrays
                           in order: mu_x, then sxx, then maf. NOT interleaved.
    gidx   [gidx_offset]   int64 x n_variants (present iff HAS_GIDX)
    data   [data_offset]   n_variants x bytes_per_variant

Section order as written by this module (and by the production converter) is
header, stats, gidx, data, so data_offset = 256 + 12*n_variants + 8*n_variants.
A reader must honour the offsets rather than assume that layout.

2-BIT ENCODING - big-endian WITHIN each byte
--------------------------------------------
Sample i lives in byte i>>2 at bit shift 6-2*(i&3), so sample 0 is the HIGH
pair. Values 0/1/2 are dosages; 3 is missing. bytes_per_variant = ceil(n/4),
and trailing bits in the last byte are zero (which decodes as dosage 0, not
missing - readers must bound loops by n_samples).

HAP2BIT ENCODING (phased) - THE SAME BYTES, READ DIFFERENTLY
-------------------------------------------------------------
A phased biallelic genotype is two 1-bit alleles, which is exactly the width of
one 2-bit dosage field, so a phased variant record has the SAME
bytes_per_variant = ceil(n_samples/4) and needs no new layout.

Pack allele bits 8 per byte, MSB first: haplotype j lives in byte j>>3 at bit
shift 7-(j&7). Substituting j = 2i and j = 2i+1 gives byte i>>2 at shifts
7-2*(i&3) and 6-2*(i&3) -- which are precisely the high and low bits of sample
i's existing 2-bit field. The haplotype-indexed and sample-indexed views are
therefore THE SAME BYTES, and both are available without conversion: an HMM
reads 1 bit per haplotype, association code reads 2 bits per sample.

    sample i's code = (hap_{2i} << 1) | hap_{2i+1}      hap_{2i} is the HIGH bit

All four codes are meaningful, so there is no spare code for missing -- which is
correct, since phased reference panels are required to be non-missing. A
HAP2BIT file must never set HAS_MISSING.

THE TRAP: THE TWO ENCODINGS SHARE BYTES BUT NOT MEANING
--------------------------------------------------------
    code   unphased (2bit)      phased (hap2bit)
    00     dosage 0             0|0  -> dosage 0     same
    01     dosage 1             0|1  -> dosage 1     same
    10     dosage 2             1|0  -> dosage 1     DIFFERENT
    11     missing              1|1  -> dosage 2     DIFFERENT

Reading a phased file as unphased does not fail, does not produce NaN, and does
not look wrong: it silently reports the wrong dosage for every het-on-the-first-
haplotype and every homozygous-alt call. Because the two agree on codes 00 and
01, a spot check of low-frequency variants can pass while the file is garbage.

Every dosage-returning read path therefore REFUSES a PHASED file unless the
caller opts in explicitly, at which point the popcount map above is applied.
Do not remove those guards to make a call site compile.

The stored mu_x/sxx/maf of a phased file are written under the popcount map, so
they describe dosages and remain directly comparable with an unphased file of
the same cohort.

PER-VARIANT STATS ARE SAMPLE-SET-SPECIFIC
------------------------------------------
mu_x, sxx and maf are computed over THIS file's samples, over VALID genotypes
only -- finite and within [0, 2], which is the production converter's exact
definition (pgen_to_cugen.py compute_variant_stats). Subsetting samples
invalidates them - recompute (cg.subset /
cg.repair) rather than copying them across. This is the single most common way
to corrupt a .cugen silently: stale stats make beta = num/sxx meaningless while
every structural check still passes.
"""
import os
import struct

import numpy as np

CUGEN_MAGIC = b"CUPGEN01"
CUGEN_VERSION = 1
HEADER_SIZE = 256

ENCODING_2BIT = 0
ENCODING_UINT8 = 1
ENCODING_FLOAT16 = 2
ENCODING_FLOAT32 = 3
ENCODING_HAP2BIT = 4

FLAG_HAS_MISSING = 1
FLAG_HAS_GIDX_MAP = 2
FLAG_PHASED = 4

# hap2bit stores two 1-bit alleles where 2bit stores one dosage, so the two
# encodings have identical bytes_per_variant. See the module docstring: this is
# a coincidence of width that the whole phased design rests on, not an accident
# to be tidied away.
_ENC_BYTES = {ENCODING_2BIT: lambda n: (n + 3) // 4,
              ENCODING_UINT8: lambda n: n,
              ENCODING_FLOAT16: lambda n: n * 2,
              ENCODING_FLOAT32: lambda n: n * 4,
              ENCODING_HAP2BIT: lambda n: (n + 3) // 4}


def pack_2bit(geno_u8):
    """Pack uint8 genotypes (0,1,2,3) into 2-bit, big-endian within byte."""
    from ._codec import pack_cugen
    return pack_cugen(np.asarray(geno_u8, dtype=np.uint8).reshape(1, -1))[0]


def unpack_2bit(packed, n_samples):
    """Inverse of pack_2bit; returns uint8 with 3 = missing."""
    from ._codec import unpack_cugen
    return unpack_cugen(np.frombuffer(packed, dtype=np.uint8).reshape(1, -1), n_samples)[0]


def pack_hap2bit(alleles):
    """Pack 1-bit phased alleles into bytes, MSB first, 8 haplotypes per byte.

    `alleles` is (2*n_samples,) with haplotype j at index j, or (n_samples, 2)
    with column 0 the first haplotype. Values must be 0 or 1.

    Haplotype j lands in byte j>>3 at shift 7-(j&7), which for j = 2i and 2i+1
    coincides with the high and low bits of sample i's 2-bit field -- so this
    produces byte-for-byte the same output as pack_2bit would on the derived
    codes (hap_{2i} << 1) | hap_{2i+1}. test_bit_views_coincide pins that.
    """
    a = np.asarray(alleles)
    if a.ndim == 2:
        if a.shape[1] != 2:
            raise ValueError(f"2-D alleles must be (n_samples, 2), got {a.shape}")
        a = a.reshape(-1)
    a = a.astype(np.uint8, copy=False)
    bad = (a > 1)
    if bad.any():
        raise ValueError(
            f"phased alleles must be 0 or 1; found {int(a[bad][0])} at "
            f"haplotype {int(np.flatnonzero(bad)[0])}. Missing calls cannot be "
            f"represented in a phased .cugen -- all four 2-bit codes are taken.")
    # np.packbits is MSB-first, which is exactly this format's bit order, and
    # it is one C call against an 8-iteration Python loop -- measured 27x
    # faster, and the two are bit-identical (test_bit_views_coincide covers
    # every awkward length). It also pads with zeros itself.
    return np.packbits(a)


def unpack_hap2bit(packed, n_haplotypes):
    """Inverse of pack_hap2bit; returns uint8 0/1 of length n_haplotypes.

    Note the length is in HAPLOTYPES (2 * n_samples), not samples. Passing
    n_samples here silently returns the first half of the cohort.
    """
    b = np.frombuffer(packed, dtype=np.uint8)
    return np.unpackbits(b)[:n_haplotypes]


def hap2bit_dosages(packed, n_samples):
    """Phased packed bytes -> dosages, via the popcount map (see docstring).

    This is the ONLY correct way to read a HAP2BIT record as dosages;
    unpack_2bit on the same bytes maps 10 -> 2 and 11 -> missing, both wrong.
    """
    h = unpack_hap2bit(packed, 2 * n_samples)
    return (h[0::2] + h[1::2]).astype(np.uint8)


def variant_stats(dosages):
    """mu_x, sxx, maf over VALID samples (finite, in [0, 2]).

    sxx is the centred sum of squares of the non-missing dosages, which is what
    the association code divides by; maf is folded to <= 0.5.
    """
    d = np.asarray(dosages, dtype=np.float64)
    # Valid mask matches the production converter (pgen_to_cugen.py
    # compute_variant_stats) EXACTLY: finite and within [0, 2]. That excludes
    # the 2-bit missing code 3, but also excludes out-of-range dosages, which
    # `d != 3` would silently have let through for the float encodings.
    ok = np.isfinite(d) & (d >= 0) & (d <= 2)
    n = int(ok.sum())
    if n == 0:
        return 0.0, 0.0, 0.0, True
    x = d[ok]
    mu = float(x.mean())
    sxx = float(((x - mu) ** 2).sum())
    af = mu / 2.0
    maf = float(min(af, 1.0 - af))
    return mu, sxx, maf, bool(n < d.size)


class CugenWriter:
    """Streaming .cugen writer.

    Usage:
        with CugenWriter(path, n_samples, n_variants) as w:
            for gidx, dosages in source:
                w.add_variant(gidx, dosages)

    Variants must be added in the order they should appear. Stats, gidx and the
    header are written on close, so the file is only valid after the context
    manager exits.
    """

    def __init__(self, path, n_samples, n_variants, encoding=ENCODING_2BIT):
        if encoding not in _ENC_BYTES:
            raise ValueError(f"unknown encoding {encoding}")
        self.path = str(path)
        self.n_samples = int(n_samples)
        self.n_variants = int(n_variants)
        self.encoding = int(encoding)
        self.bytes_per_variant = _ENC_BYTES[encoding](self.n_samples)

        self.stats_offset = HEADER_SIZE
        self.stats_size = self.n_variants * 4 * 3
        self.gidx_offset = self.stats_offset + self.stats_size
        self.gidx_size = self.n_variants * 8
        self.data_offset = self.gidx_offset + self.gidx_size

        self.mu_x = np.zeros(self.n_variants, dtype=np.float32)
        self.sxx = np.zeros(self.n_variants, dtype=np.float32)
        self.maf = np.zeros(self.n_variants, dtype=np.float32)
        self.gidx = np.zeros(self.n_variants, dtype=np.int64)
        self.flags = FLAG_HAS_GIDX_MAP
        if self.encoding == ENCODING_HAP2BIT:
            self.flags |= FLAG_PHASED
        self.i = 0
        self.f = None

    def __enter__(self):
        self.f = open(self.path, "wb")
        self.f.write(b"\x00" * (HEADER_SIZE + self.stats_size + self.gidx_size))
        return self

    def add_variant(self, gidx, dosages):
        if self.encoding == ENCODING_HAP2BIT:
            raise ValueError(
                "this writer is HAP2BIT (phased); use add_variant_phased(gidx, "
                "alleles) with 0/1 alleles. add_variant takes DOSAGES, and "
                "dosages cannot be written to a phased file without inventing "
                "a phase that was never observed.")
        if self.i >= self.n_variants:
            raise IndexError(f"more than the declared {self.n_variants} variants")
        d = np.asarray(dosages)
        if d.size != self.n_samples:
            raise ValueError(f"variant {self.i}: {d.size} dosages != "
                             f"n_samples {self.n_samples}")
        mu, sxx, maf, has_missing = variant_stats(d)
        self.mu_x[self.i] = mu
        self.sxx[self.i] = sxx
        self.maf[self.i] = maf
        self.gidx[self.i] = int(gidx)
        if has_missing:
            self.flags |= FLAG_HAS_MISSING

        if self.encoding == ENCODING_2BIT:
            g = np.asarray(d, dtype=np.float64)
            u = np.where(np.isfinite(g), np.rint(np.nan_to_num(g, nan=3.0)), 3.0)
            u = np.clip(u, 0, 3).astype(np.uint8)
            self.f.write(pack_2bit(u).tobytes())
        elif self.encoding == ENCODING_UINT8:
            u = np.clip(np.nan_to_num(np.asarray(d), nan=3.0), 0, 3).astype(np.uint8)
            self.f.write(u.tobytes())
        elif self.encoding == ENCODING_FLOAT16:
            self.f.write(np.asarray(d, dtype=np.float16).tobytes())
        else:
            self.f.write(np.asarray(d, dtype=np.float32).tobytes())
        self.i += 1

    def add_variant_phased(self, gidx, alleles):
        """Add one phased variant. `alleles` is (2*n_samples,) or (n_samples, 2).

        Stats are recorded under the popcount map so mu_x/sxx/maf describe
        dosages and stay comparable with an unphased file of the same cohort.
        HAS_MISSING is never set: a phased record has no code for missing, and
        pack_hap2bit rejects anything that is not 0 or 1.
        """
        if self.encoding != ENCODING_HAP2BIT:
            raise ValueError(
                f"add_variant_phased needs encoding=ENCODING_HAP2BIT, this "
                f"writer has encoding {self.encoding}. Writing alleles into a "
                f"dosage encoding would store 1|0 as dosage 1 and lose phase "
                f"silently.")
        if self.i >= self.n_variants:
            raise IndexError(f"more than the declared {self.n_variants} variants")
        a = np.asarray(alleles)
        n_hap = a.size if a.ndim == 1 else a.shape[0] * a.shape[1]
        if n_hap != 2 * self.n_samples:
            raise ValueError(f"variant {self.i}: {n_hap} alleles != "
                             f"2 * n_samples {2 * self.n_samples}")
        packed = pack_hap2bit(a)
        # Dosages come straight from the alleles. Deriving them by unpacking
        # what was just packed round-trips the data for nothing and measured a
        # full minute over chromosome 20.
        flat = a.reshape(-1)
        mu, sxx, maf, _ = variant_stats(
            (flat[0::2].astype(np.int16) + flat[1::2]).astype(np.int16))
        self.mu_x[self.i] = mu
        self.sxx[self.i] = sxx
        self.maf[self.i] = maf
        self.gidx[self.i] = int(gidx)
        self.f.write(packed.tobytes())
        self.i += 1

    def add_variants_bulk(self, gidx, dosages):
        """Append many variants at once. `dosages` is (n_samples, n_variants).

        Float encodings store the values verbatim, so the whole block can be
        transposed and written in one call and the per-variant statistics can be
        computed with array operations. The per-variant loop costs a Python
        iteration and several small numpy calls per marker, which is 32 seconds
        for one chromosome of imputed output -- more than the entire GPU
        computation that produced it.
        """
        if self.encoding not in (ENCODING_FLOAT16, ENCODING_FLOAT32):
            raise ValueError(
                f"add_variants_bulk is for float encodings; this writer has "
                f"encoding {self.encoding}. The integer encodings need "
                f"per-variant packing.")
        # Keep the caller's precision. Promoting float32 dosages to float64
        # doubles every temporary below, and there are four of them at full
        # size; at 1.73M markers that was the largest single phase in a
        # chromosome run.
        d = np.asarray(dosages)
        if d.dtype not in (np.float32, np.float64):
            d = d.astype(np.float64)
        if d.ndim != 2 or d.shape[0] != self.n_samples:
            raise ValueError(f"dosages must be (n_samples={self.n_samples}, "
                             f"n_variants), got {d.shape}")
        n = d.shape[1]
        if self.i + n > self.n_variants:
            raise IndexError(f"writing {n} variants at position {self.i} "
                             f"exceeds the declared {self.n_variants}")
        g = np.asarray(gidx, dtype=np.int64)
        if g.size != n:
            raise ValueError(f"{g.size} gidx entries for {n} variants")

        ok = np.isfinite(d) & (d >= 0) & (d <= 2)          # same rule as
        cnt = ok.sum(axis=0)                                # variant_stats()
        # Accumulate in float64 from float32 inputs without materialising a
        # float64 copy of the whole block: sum and sum-of-squares are enough for
        # both statistics, and np.einsum keeps the accumulator wide while the
        # operand stays narrow.
        w = np.where(ok, d, 0)
        s1 = w.sum(axis=0, dtype=np.float64)
        s2 = np.einsum("ij,ij->j", w, w, dtype=np.float64)
        mu = s1 / np.maximum(cnt, 1)
        sxx = s2 - 2.0 * mu * s1 + mu * mu * cnt
        af = mu / 2.0
        maf = np.minimum(af, 1.0 - af)
        empty = cnt == 0
        mu[empty] = 0.0; sxx[empty] = 0.0; maf[empty] = 0.0
        if bool((cnt < self.n_samples).any()):
            self.flags |= FLAG_HAS_MISSING

        sl = slice(self.i, self.i + n)
        self.mu_x[sl] = mu
        self.sxx[sl] = sxx
        self.maf[sl] = maf
        self.gidx[sl] = g
        dt = np.float16 if self.encoding == ENCODING_FLOAT16 else np.float32
        self.f.write(np.ascontiguousarray(d.T, dtype=dt).tobytes())
        self.i += n

    def _finalize(self):
        if self.i != self.n_variants:
            raise ValueError(f"declared {self.n_variants} variants, wrote {self.i}")
        self.f.seek(self.stats_offset)
        self.f.write(self.mu_x.tobytes())
        self.f.write(self.sxx.tobytes())
        self.f.write(self.maf.tobytes())
        self.f.seek(self.gidx_offset)
        self.f.write(self.gidx.tobytes())

        h = bytearray(HEADER_SIZE)
        h[0:8] = CUGEN_MAGIC
        struct.pack_into("<I", h, 8, CUGEN_VERSION)
        struct.pack_into("<I", h, 12, self.encoding)
        struct.pack_into("<Q", h, 16, self.n_samples)
        struct.pack_into("<Q", h, 24, self.n_variants)
        struct.pack_into("<Q", h, 32, self.bytes_per_variant)
        struct.pack_into("<Q", h, 40, self.stats_offset)
        struct.pack_into("<Q", h, 48, self.data_offset)
        struct.pack_into("<Q", h, 56, self.gidx_offset)
        struct.pack_into("<I", h, 64, self.flags)
        self.f.seek(0)
        self.f.write(bytes(h))

    def __exit__(self, exc_type, exc, tb):
        if self.f:
            if exc_type is None:
                self._finalize()
            self.f.close()
            self.f = None
        return False


def write_cugen(path, dosages, gidx=None, encoding=ENCODING_2BIT):
    """Write a whole (n_samples, n_variants) dosage array to `path`."""
    a = np.asarray(dosages)
    if a.ndim != 2:
        raise ValueError("dosages must be 2-D (n_samples, n_variants)")
    n_samples, n_variants = a.shape
    g = np.arange(n_variants, dtype=np.int64) if gidx is None else \
        np.asarray(gidx, dtype=np.int64)
    if g.size != n_variants:
        raise ValueError(f"gidx has {g.size} entries, need {n_variants}")
    with CugenWriter(path, n_samples, n_variants, encoding=encoding) as w:
        for j in range(n_variants):
            w.add_variant(int(g[j]), a[:, j])
    return path


def write_cugen_phased(path, alleles, gidx=None):
    """Write a phased haplotype array to `path` as a HAP2BIT .cugen.

    `alleles` is (2*n_samples, n_variants) of 0/1, haplotype-major on the row
    axis: rows 2i and 2i+1 are sample i's two haplotypes. That row order is the
    format's, not a convention of this function -- see the module docstring.
    """
    a = np.asarray(alleles)
    if a.ndim != 2:
        raise ValueError("alleles must be 2-D (2*n_samples, n_variants)")
    n_hap, n_variants = a.shape
    if n_hap % 2:
        raise ValueError(f"{n_hap} haplotype rows is odd; expected 2*n_samples")
    n_samples = n_hap // 2
    g = np.arange(n_variants, dtype=np.int64) if gidx is None else \
        np.asarray(gidx, dtype=np.int64)
    if g.size != n_variants:
        raise ValueError(f"gidx has {g.size} entries, need {n_variants}")
    with CugenWriter(path, n_samples, n_variants,
                     encoding=ENCODING_HAP2BIT) as w:
        for j in range(n_variants):
            w.add_variant_phased(int(g[j]), a[:, j])
    return path


def validate_cugen(path, verbose=True):
    """Structural check. Returns a dict; raises nothing, reports `ok`.

    NOTE this validates STRUCTURE only. It cannot tell you whether the stored
    stats match the stored genotypes -- for that use cg.repair(dry_run=True),
    which recomputes them. A file with stale stats passes every check here.
    """
    r = {"path": str(path), "ok": False, "problems": []}
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        h = f.read(HEADER_SIZE)
    if len(h) < HEADER_SIZE:
        r["problems"].append("file shorter than a header")
        return r
    if h[0:8] != CUGEN_MAGIC:
        r["problems"].append(f"bad magic {h[0:8]!r}")
    g = lambda fmt, off: struct.unpack_from(fmt, h, off)[0]      # noqa: E731
    r.update(version=g("<I", 8), encoding=g("<I", 12),
             n_samples=g("<Q", 16), n_variants=g("<Q", 24),
             bytes_per_variant=g("<Q", 32), stats_offset=g("<Q", 40),
             data_offset=g("<Q", 48), gidx_offset=g("<Q", 56), flags=g("<I", 64))
    if r["version"] != CUGEN_VERSION:
        r["problems"].append(f"version {r['version']} != {CUGEN_VERSION}")
    exp_bpv = _ENC_BYTES.get(r["encoding"], lambda n: -1)(r["n_samples"])
    if r["bytes_per_variant"] != exp_bpv:
        r["problems"].append(
            f"bytes_per_variant {r['bytes_per_variant']} != {exp_bpv} "
            f"for encoding {r['encoding']} at n_samples {r['n_samples']}")
    exp_size = r["data_offset"] + r["n_variants"] * r["bytes_per_variant"]
    if size != exp_size:
        r["problems"].append(f"file size {size} != expected {exp_size}")
    if r["stats_offset"] < HEADER_SIZE:
        r["problems"].append("stats_offset inside the header")
    if r["data_offset"] < r["stats_offset"] + 12 * r["n_variants"]:
        r["problems"].append("data_offset overlaps the stats block")
    r["has_gidx"] = bool(r["flags"] & FLAG_HAS_GIDX_MAP)
    r["has_missing"] = bool(r["flags"] & FLAG_HAS_MISSING)
    r["phased"] = bool(r["flags"] & FLAG_PHASED)

    # PHASED and HAP2BIT must agree. They carry the same bytes_per_variant as
    # ENCODING_2BIT, so a file with one set and not the other passes every size
    # check above while decoding to the wrong dosages -- exactly the failure the
    # encoding table in the module docstring describes.
    if r["phased"] and r["encoding"] != ENCODING_HAP2BIT:
        r["problems"].append(
            f"PHASED flag set but encoding is {r['encoding']}, not "
            f"{ENCODING_HAP2BIT} (hap2bit)")
    if r["encoding"] == ENCODING_HAP2BIT and not r["phased"]:
        r["problems"].append("encoding is hap2bit but the PHASED flag is unset")
    if r["phased"] and r["has_missing"]:
        r["problems"].append(
            "PHASED and HAS_MISSING are both set; a phased record has no code "
            "for missing (all four 2-bit codes are valid genotypes)")

    r["ok"] = not r["problems"]
    if verbose:
        print(f"{'OK  ' if r['ok'] else 'BAD '} {path}")
        print(f"     n_samples={r['n_samples']:,} n_variants={r['n_variants']:,} "
              f"enc={r['encoding']} bpv={r['bytes_per_variant']:,} "
              f"gidx={r['has_gidx']} missing={r['has_missing']} "
              f"phased={r['phased']}")
        for p in r["problems"]:
            print(f"     - {p}")
    return r
