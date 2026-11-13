"""One 2-bit genotype codec for the library: cugen 2-bit, cugen hap2bit and PLINK 1 .bed.

A byte holds four samples in all three layouts; they differ in bit order and in what a
2-bit code means:

* **cugen 2-bit** (unphased): sample 0 in the HIGH bits (7-6). Code = count of the
  file's counted allele (0, 1, 2); 3 = missing.
* **cugen hap2bit** (phased): same byte order; the two bits are the sample's two
  haplotype alleles, so the dosage is their popcount (00 -> 0, 01 -> 1, 10 -> 1,
  11 -> 2). There is no missing code.
* **PLINK 1 .bed** (SNP-major): sample 0 in the LOW bits (1-0). 00 = hom A1,
  01 = missing, 10 = het, 11 = hom A2, where A1 / A2 are .bim columns 5 / 6.

Decoding is one ``(256, 4)`` byte lookup table per layout (:data:`CUGEN_LUT`,
:data:`HAP2BIT_LUT`, :func:`bed_lut`), indexed by the packed bytes and reshaped; the
same expression runs on numpy and cupy arrays (:func:`decode`). Decoded values are
``uint8`` counts of the counted allele with 3 for missing.

The ``counted`` vocabulary for .bed data is shared library-wide: ``"A1"`` counts .bim
column 5 (PLINK / REGENIE convention), ``"A2"`` counts column 6.
"""
from __future__ import annotations

import numpy as np

MISSING = 3
BED_MAGIC = b"\x6c\x1b\x01"
COUNTED_CHOICES = ("A1", "A2")


def byte_lut(code_map, shifts):
    """(256, 4) uint8: entry [b, k] = code_map[(b >> shifts[k]) & 3]."""
    b = np.arange(256, dtype=np.uint8)
    cm = np.asarray(code_map, dtype=np.uint8)
    return np.stack([cm[(b >> s) & 3] for s in shifts], 1)


#: cugen 2-bit: codes are counts already; sample 0 in the high bits.
CUGEN_LUT = byte_lut([0, 1, 2, MISSING], (6, 4, 2, 0))
#: cugen hap2bit (phased): popcount of the two allele bits; sample 0 in the high bits.
HAP2BIT_LUT = byte_lut([0, 1, 1, 2], (6, 4, 2, 0))
# PLINK .bed: 00 hom A1, 01 missing, 10 het, 11 hom A2; sample 0 in the low bits.
_BED_CODE = {"A1": [2, MISSING, 1, 0], "A2": [0, MISSING, 1, 2]}
_BED_LUTS = {k: byte_lut(v, (0, 2, 4, 6)) for k, v in _BED_CODE.items()}


def check_counted(counted):
    """Validate the shared ``counted`` vocabulary ("A1" | "A2")."""
    if counted not in COUNTED_CHOICES:
        raise ValueError(f"counted must be 'A1' (bim column 5) or 'A2' (bim column 6), got {counted!r}")
    return counted


def bed_lut(counted="A1"):
    """(256, 4) .bed byte table counting ``counted`` ("A1" = bim column 5, "A2" = column 6)."""
    return _BED_LUTS[check_counted(counted)]


def bytes_per_variant(n):
    return (int(n) + 3) // 4


def decode(packed, lut, n, xp=np):
    """packed (V, bpv) uint8 -> (V, n) uint8 through a (256, 4) table (numpy or cupy)."""
    p = xp.asarray(packed)
    if p.ndim == 1:
        p = p.reshape(1, -1)
    return lut[p].reshape(p.shape[0], -1)[:, :n]


def unpack_cugen(packed, n):
    """cugen 2-bit records (V, bpv) -> (V, n) counts, 3 = missing."""
    return decode(packed, CUGEN_LUT, n)


def unpack_hap2bit(packed, n):
    """cugen hap2bit records (V, bpv) -> (V, n) dosages (popcount map, no missing)."""
    return decode(packed, HAP2BIT_LUT, n)


def unpack_bed(packed, n, counted="A1"):
    """.bed records (V, bpv) -> (V, n) counts of ``counted``, 3 = missing."""
    return decode(packed, bed_lut(counted), n)


def _pack(codes, shifts):
    c = np.asarray(codes, dtype=np.uint8)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    v, n = c.shape
    pad = (-n) % 4
    if pad:
        c = np.concatenate([c, np.zeros((v, pad), dtype=np.uint8)], axis=1)
    r = c.reshape(v, -1, 4)
    return ((r[:, :, 0] << shifts[0]) | (r[:, :, 1] << shifts[1]) | (r[:, :, 2] << shifts[2])
            | (r[:, :, 3] << shifts[3])).astype(np.uint8)


def pack_cugen(codes):
    """(V, N) uint8 codes 0..3 -> (V, ceil(N/4)) cugen 2-bit bytes (sample 0 in the high bits)."""
    return _pack(codes, (6, 4, 2, 0))


def pack_bed(counts, counted="A1"):
    """(V, N) counts of ``counted`` (0..2, 3 = missing) -> (V, ceil(N/4)) .bed bytes."""
    c = np.asarray(counts)
    if c.ndim == 1:
        c = c.reshape(1, -1)
    if c.size and int(c.max()) > MISSING:
        raise ValueError("counts must be 0, 1, 2 or 3 (missing)")
    # inverse of the decode table: count -> 2-bit .bed code
    inv = np.zeros(4, dtype=np.uint8)
    for code, val in enumerate(_BED_CODE[check_counted(counted)]):
        inv[val] = code
    return _pack(inv[c.astype(np.intp)], (0, 2, 4, 6))
