"""Genotype block readers for PLINK .bed and .cugen files (CPU or GPU decode, one code path).

These are the inputs of the background engines in :mod:`cugen.background` and of the
blocked step-4 engine. Blocks are read with positioned reads into owned buffers
(:func:`cugen.io._pread_rows`, shared with :class:`cugen.io.CugenReader`), not through
a memory map, so the process RSS reflects only what the methods hold.

Allele convention (explicit, reported in every output)
------------------------------------------------------
* :class:`BedGenotypes` counts ``counted="A1"`` (``.bim`` column 5) by default, as PLINK
  and REGENIE do; ``counted="A2"`` counts column 6. (plink2 writes ALT in column 5.)
* :class:`CugenGenotypes` counts the cugen's stored allele: the ``ALT`` column of the
  cugen annotation. :func:`cugen.bed2cugen` stores ``.bim`` column 6 by default, so a
  .bed read with ``counted="A1"`` and its default cugen give opposite signs.

Decoded blocks are ``uint8`` arrays ``(variants, samples)`` holding the count of the
counted allele (0, 1, 2) and 3 for missing, produced by one byte lookup table per format
(:mod:`cugen._codec`) followed by the sample gather; the same ``xp`` code runs on numpy
and cupy. Phased (hap2bit) cugens are decoded to dosages with the popcount map.

Variant selection (``snps=``) accepts a list of IDs, a path to a SNP list / PLINK
``.prune.in`` file (first whitespace token per line), a boolean mask or integer rows;
``chromosomes=`` restricts it to a set of chromosomes.
"""
from __future__ import annotations

import os
import queue
import re
import threading

import numpy as np

from . import _codec
from .io import _pread_rows

__all__ = ["GenotypeSource", "BedGenotypes", "CugenGenotypes", "open_genotypes"]

BED_MAGIC = _codec.BED_MAGIC
MISSING = _codec.MISSING
_COALESCE_GAP = 64           # rows: nearby wanted rows are fetched in one read


def _read_list(snps):
    if isinstance(snps, (str, os.PathLike)):
        with open(snps) as fh:
            return [ln.split()[0] for ln in fh if ln.strip()]
    return list(snps)


class GenotypeSource:
    """Common interface. Subclasses set: ``n``, ``n_variants``, ``chrom`` (int64), ``pos``,
    ``gidx``, ``sample_ids``, ``counted`` / ``counted_label``, ``lut``, ``n_file`` and
    implement ``_read_rows(rows) -> (len(rows), bpv)`` uint8 packed bytes."""

    lut = None
    n_file = None
    samp = None              # sample gather (None = identity)
    counted = None
    counted_label = None

    # ---------------------------------------------------------------- variant metadata
    def variant_ids(self, rows=None):
        raise NotImplementedError

    def alleles(self, rows=None):
        """(counted, other) allele letters as pyarrow arrays (or ``(None, None)``)."""
        return None, None

    def stored_stats(self, rows):
        """(mu, sxx, maf) float32 over valid calls for the analysed samples, or None."""
        return None

    def select(self, snps=None, chromosomes=None):
        """Sorted global variant rows for a SNP list / path / mask / rows, on ``chromosomes``.

        Variants on non-numeric chromosomes (X, Y, MT, ... coded 0) are never selected."""
        if snps is None:
            sel = np.ones(self.n_variants, dtype=bool)
        elif isinstance(snps, np.ndarray) and snps.dtype == bool:
            if len(snps) != self.n_variants:
                raise ValueError("boolean snps mask has the wrong length")
            sel = snps.copy()
        elif isinstance(snps, np.ndarray) and np.issubdtype(snps.dtype, np.integer):
            sel = np.zeros(self.n_variants, dtype=bool)
            sel[snps] = True
        else:
            import pyarrow as pa
            import pyarrow.compute as pc
            want = _read_list(snps)
            ids = self.variant_ids()
            if ids is None:
                raise ValueError("selecting SNPs by ID needs variant IDs (pass annotation= for cugens)")
            sel = pc.is_in(ids, value_set=pa.array(want, type=pa.string())).to_numpy(zero_copy_only=False)
            sel = np.asarray(sel, dtype=bool)
            if not sel.any():
                raise ValueError(f"none of the {len(want):,} listed SNP IDs are in the genotype data")
        if chromosomes is not None:
            sel &= np.isin(self.chrom, np.asarray(list(chromosomes), dtype=np.int64))
        sel &= self.chrom > 0
        rows = np.flatnonzero(sel)
        if np.any(np.diff(self.chrom[rows]) < 0):
            raise ValueError("variants are not sorted by chromosome")
        return rows

    def chrom_ranges(self, rows):
        """[(chrom, i0, i1)] ranges of positions in `rows` per chromosome."""
        c = self.chrom[rows]
        if len(c) == 0:
            return []
        b = np.flatnonzero(np.diff(c)) + 1
        st = np.r_[0, b]
        en = np.r_[b, len(c)]
        return [(int(c[s]), int(s), int(e)) for s, e in zip(st, en)]

    def blocks(self, rows, block_size):
        """[(chrom, i0, i1)] blocks of at most block_size positions in `rows`, not crossing
        chromosome boundaries (REGENIE's block layout when block_size = bsize)."""
        out = []
        for c, s0, e0 in self.chrom_ranges(rows):
            for s in range(s0, e0, block_size):
                out.append((c, s, min(s + block_size, e0)))
        return out

    # ---------------------------------------------------------------- reading + decoding
    def read_packed(self, rows):
        rows = np.asarray(rows, dtype=np.int64)
        return self._read_rows(rows)

    def decode(self, packed, dev):
        """packed (b, bpv) uint8 -> counts (b, n) uint8 on dev (0..2 counted allele, 3 missing)."""
        xp = dev.xp
        p = xp.asarray(packed)
        codes = self._lut_on(dev)[p].reshape(p.shape[0], -1)
        if self.samp is not None:
            return codes[:, self._samp_on(dev)]
        return codes[:, :self.n_file]

    def _cached(self, key, make):
        c = self.__dict__.setdefault("_cache", {})
        if key not in c:
            c[key] = make()
        return c[key]

    def _lut_on(self, dev):
        return self._cached(("lut", str(dev)), lambda: dev.asarray(self.lut))

    def _samp_on(self, dev):
        return self._cached(("samp", str(dev)), lambda: dev.asarray(self.samp))

    def iter_blocks(self, rows, blocks, dev, prefetch=2):
        """Yield (chrom, i0, i1, counts) for each block, reading the next blocks in a thread."""
        if prefetch <= 0:
            for c, s, e in blocks:
                yield c, s, e, self.decode(self.read_packed(rows[s:e]), dev)
            return
        q = queue.Queue(maxsize=prefetch)
        stop = threading.Event()
        err = []

        def reader():
            try:
                for blk in blocks:
                    if stop.is_set():
                        break
                    c, s, e = blk
                    q.put((blk, self.read_packed(rows[s:e])))
            except BaseException as ex:  # noqa: BLE001 - re-raised in the consumer
                err.append(ex)
            finally:
                q.put(None)

        th = threading.Thread(target=reader, daemon=True)
        th.start()
        try:
            while True:
                it = q.get()
                if it is None:
                    break
                (c, s, e), pk = it
                yield c, s, e, self.decode(pk, dev)
        finally:
            stop.set()
            while th.is_alive():
                try:
                    q.get(timeout=0.1)
                except queue.Empty:
                    pass
            th.join()
        if err:
            raise err[0]

    def read_counts(self, rows=None, dev=None):
        """Decoded counts (len(rows), n) as numpy; convenient for small data and tests."""
        from ._device import device as _device
        dev = _device("cpu") if dev is None else _device(dev)
        rows = np.arange(self.n_variants) if rows is None else np.asarray(rows)
        with dev:
            return dev.asnumpy(self.decode(self.read_packed(rows), dev))

    def resident_bytes(self):
        """Approximate bytes this reader keeps resident (per-variant metadata, IDs, alleles)."""
        tot = 0
        for v in vars(self).values():
            if isinstance(v, np.ndarray):
                tot += v.nbytes
            elif hasattr(v, "nbytes") and hasattr(v, "num_rows"):      # pyarrow Table
                tot += v.nbytes
        return int(tot)

    def describe(self):
        return (f"{type(self).__name__}: {self.n:,} samples x {self.n_variants:,} variants, "
                f"counting {self.counted_label}")


# =============================================================================== PLINK bed
def _read_bim(path):
    """chr labels, pos, id, a1, a2 (ids/alleles as pyarrow string arrays)."""
    import pyarrow as pa
    import pyarrow.csv as pcsv
    names = ["chr", "id", "cm", "pos", "a1", "a2"]
    types = {"chr": pa.string(), "id": pa.string(), "cm": pa.float64(), "pos": pa.int64(),
             "a1": pa.string(), "a2": pa.string()}
    try:
        t = pcsv.read_csv(path, read_options=pcsv.ReadOptions(column_names=names, block_size=1 << 26),
                          parse_options=pcsv.ParseOptions(delimiter="\t"),
                          convert_options=pcsv.ConvertOptions(column_types=types, include_columns=
                                                              ["chr", "id", "pos", "a1", "a2"]))
    except (pa.ArrowInvalid, pa.ArrowTypeError):
        import pandas as pd
        df = pd.read_csv(path, sep=r"\s+", header=None, names=names, dtype={"chr": str, "id": str,
                                                                            "a1": str, "a2": str})
        t = pa.Table.from_pandas(df[["chr", "id", "pos", "a1", "a2"]], preserve_index=False)
    return t


def _chrom_codes(labels):
    """'1'/'chr1' -> 1; non-numeric labels (X, Y, MT, ...) -> 0 (excluded from analyses)."""
    import pyarrow as pa
    import pyarrow.compute as pc
    s = pc.replace_substring_regex(labels, pattern="^chr", replacement="")
    isnum = pc.match_substring_regex(s, pattern=r"^[0-9]+$")
    num = pc.if_else(isnum, s, pa.scalar("0"))
    return np.asarray(pc.cast(num, pa.int64()).to_numpy(zero_copy_only=False), dtype=np.int64)


class BedGenotypes(GenotypeSource):
    """PLINK 1 binary (``prefix.bed/.bim/.fam``) with a sample gather.

    samples : IDs (``id_col`` of the .fam, "IID" or "FID") in the order wanted; None = all.
    counted : "A1" (default; .bim column 5, the PLINK/REGENIE allele) or "A2" (column 6, the
              allele :func:`cugen.bed2cugen` stores by default).
    """

    def __init__(self, prefix, samples=None, id_col="IID", counted="A1"):
        import pandas as pd
        _codec.check_counted(counted)
        if id_col not in ("IID", "FID"):
            raise ValueError("id_col must be 'IID' or 'FID'")
        self.prefix = str(prefix)
        self.counted = counted
        self.counted_label = f"bim {counted} (column {5 if counted == 'A1' else 6})"
        fam = pd.read_csv(self.prefix + ".fam", sep=r"\s+", header=None, dtype=str, usecols=[0, 1])
        fam_ids = fam[{"FID": 0, "IID": 1}[id_col]].to_numpy()
        self.n_file = len(fam_ids)
        if samples is None:
            self.samp = None
            self.sample_ids = fam_ids.astype(str)
        else:
            pos = {f: i for i, f in enumerate(fam_ids)}
            samples = [str(s) for s in samples]
            miss = [s for s in samples if s not in pos]
            if miss:
                raise KeyError(f"{len(miss)} requested samples not in the .fam (e.g. {miss[:3]})")
            self.samp = np.array([pos[s] for s in samples], dtype=np.int64)
            self.sample_ids = np.asarray(samples, dtype=str)
            if len(self.samp) == self.n_file and np.array_equal(self.samp, np.arange(self.n_file)):
                self.samp = None
        self.n = len(self.sample_ids)
        self.sample_order_is_file_order = self.samp is None or bool(np.all(np.diff(self.samp) > 0))
        self._bim = _read_bim(self.prefix + ".bim")
        self.n_variants = self._bim.num_rows
        self.chrom = _chrom_codes(self._bim.column("chr"))
        self.pos = self._bim.column("pos").to_numpy()
        self.gidx = np.arange(self.n_variants, dtype=np.int64)
        self.bpv = _codec.bytes_per_variant(self.n_file)
        size = os.path.getsize(self.prefix + ".bed")
        if size != 3 + self.n_variants * self.bpv:
            raise ValueError(f"{self.prefix}.bed has {size} bytes, expected 3 + {self.n_variants} x {self.bpv}")
        with open(self.prefix + ".bed", "rb") as fh:
            if fh.read(3) != BED_MAGIC:
                raise ValueError("not a variant-major PLINK 1 .bed file")
        self._fd = os.open(self.prefix + ".bed", os.O_RDONLY)
        self.lut = _codec.bed_lut(counted)

    def _read_rows(self, rows):
        return _pread_rows(self._fd, 3, self.bpv, rows, max_gap=_COALESCE_GAP, path=self.prefix + ".bed")

    def __del__(self):
        try:
            os.close(self._fd)
        except (AttributeError, OSError):
            pass

    def variant_ids(self, rows=None):
        c = self._bim.column("id")
        return c if rows is None else c.take(rows)

    def alleles(self, rows=None):
        a1, a2 = self._bim.column("a1"), self._bim.column("a2")
        cnt, oth = (a1, a2) if self.counted == "A1" else (a2, a1)
        if rows is not None:
            cnt, oth = cnt.take(rows), oth.take(rows)
        return cnt, oth


# =============================================================================== .cugen
class CugenGenotypes(GenotypeSource):
    """Per-chromosome ``chr{c}.cugen`` files (2-bit, or phased hap2bit read as dosages).

    path        : directory with chr{c}.cugen files, or a list of files (chromosome order).
    chromosomes : optional subset of chromosomes to open (directory input).
    annotation  : optional feather/parquet/DataFrame with gidx, CHR, POS, ID, REF, ALT (gives
                  IDs for ``snps=`` selection and the output columns). ``ALT`` is the
                  counted allele.

    Cugens hold their own sample set and stored stats (mu_x, sxx, maf over valid calls);
    samples cannot be subset here (the stored stats would no longer apply).
    """

    def __init__(self, path, chromosomes=None, annotation=None):
        from .io import ENCODING_2BIT, ENCODING_HAP2BIT, CugenReader, encoding_name
        if isinstance(path, (str, os.PathLike)) and os.path.isdir(path):
            files = {}
            for f in os.listdir(path):
                m = re.fullmatch(r"chr(\d+)\.cugen", f)
                if m:
                    files[int(m.group(1))] = os.path.join(path, f)
            if chromosomes is not None:
                want = {int(c) for c in chromosomes}
                files = {c: p for c, p in files.items() if c in want}
            chrom_of_file = sorted(files)
            paths = [files[c] for c in chrom_of_file]
            self.directory = str(path)
        else:
            paths = [str(p) for p in path]
            chrom_of_file = [int(re.search(r"chr(\d+)", os.path.basename(p)).group(1)) for p in paths]
            if chromosomes is not None:
                want = {int(c) for c in chromosomes}
                keep = [i for i, c in enumerate(chrom_of_file) if c in want]
                paths = [paths[i] for i in keep]
                chrom_of_file = [chrom_of_file[i] for i in keep]
            self.directory = os.path.dirname(paths[0]) if paths else None
        if not paths:
            raise FileNotFoundError(f"no chr*.cugen files under {path}")
        self.counted = "ALT"
        self.counted_label = "cugen ALT (annotation ALT; bim column 6 for default bed2cugen files)"
        self.files = paths
        self._fds, self._data_off = [], []
        chrom, gidx, mu, sxx, maf, starts = [], [], [], [], [], [0]
        n = bpv = enc = None
        for c, p in zip(chrom_of_file, paths):
            r = CugenReader(p)
            if r.encoding not in (ENCODING_2BIT, ENCODING_HAP2BIT):
                raise ValueError(f"{p}: encoding {encoding_name(r.encoding)} is not supported; "
                                 f"CugenGenotypes reads 2bit and phased hap2bit files")
            if n is None:
                n, bpv, enc = r.n_samples, r.bytes_per_variant, r.encoding
            elif r.n_samples != n:
                raise ValueError(f"{p}: {r.n_samples} samples, other files have {n}")
            elif r.encoding != enc:
                raise ValueError(f"{p}: encoding {encoding_name(r.encoding)} differs from the other files "
                                 f"({encoding_name(enc)}); convert them to one encoding")
            chrom.append(np.full(r.n_variants, c, dtype=np.int64))
            gidx.append(r.gidx)
            mu.append(r.mu_x)
            sxx.append(r.sxx)
            maf.append(r.maf)
            self._fds.append(os.open(p, os.O_RDONLY))
            self._data_off.append(int(r.data_offset))
            starts.append(starts[-1] + r.n_variants)
            r.close()
        self.n = self.n_file = int(n)
        self.bpv = int(bpv)
        self.phased = enc == ENCODING_HAP2BIT
        self.encoding = encoding_name(enc)
        self.chrom = np.concatenate(chrom)
        self.gidx = np.concatenate(gidx)
        self._mu, self._sxx, self._maf = np.concatenate(mu), np.concatenate(sxx), np.concatenate(maf)
        self._starts = np.asarray(starts, dtype=np.int64)
        self.n_variants = int(self._starts[-1])
        # hap2bit: the two bits are allele calls, dosage = popcount; the stored stats were
        # computed on those dosages by the writer, so stats="stored" stays consistent.
        self.lut = _codec.HAP2BIT_LUT if self.phased else _codec.CUGEN_LUT
        self.samp = None
        sp = paths[0] + ".samples.txt"
        ids = None
        if os.path.exists(sp):
            with open(sp) as fh:
                ids = np.array([ln.split()[0] for ln in fh if ln.strip()], dtype=str)
        self.sample_ids = ids if ids is not None and len(ids) == self.n else np.arange(self.n).astype(str)
        self.sample_order_is_file_order = True
        self._ann = None
        self.pos = np.zeros(self.n_variants, dtype=np.int64)
        if annotation is not None:
            self._load_annotation(annotation)

    def _load_annotation(self, annotation):
        import pyarrow as pa
        if isinstance(annotation, (str, os.PathLike)):
            p = str(annotation)
            if p.endswith(".parquet"):
                import pyarrow.parquet as pq
                t = pq.read_table(p)
            else:
                import pyarrow.feather as pf
                t = pf.read_table(p)
        elif isinstance(annotation, pa.Table):
            t = annotation
        else:
            t = pa.Table.from_pandas(annotation, preserve_index=False)
        g = t.column("gidx").to_numpy()
        order = np.full(int(max(g.max(), self.gidx.max())) + 1, -1, dtype=np.int64)
        order[g] = np.arange(len(g))
        take = order[self.gidx]
        if (take < 0).any():
            raise ValueError(f"annotation lacks {(take < 0).sum():,} of the cugen gidx values")
        cols = [c for c in ("POS", "ID", "REF", "ALT") if c in t.column_names]
        self._ann = t.select(cols).take(pa.array(take))
        if "POS" in cols:
            self.pos = self._ann.column("POS").to_numpy().astype(np.int64)

    def _read_rows(self, rows):
        f = np.searchsorted(self._starts, rows, side="right") - 1
        if len(rows) and f[0] == f[-1]:
            fi = int(f[0])
            return _pread_rows(self._fds[fi], self._data_off[fi], self.bpv, rows - self._starts[fi],
                               max_gap=_COALESCE_GAP, path=self.files[fi])
        out = np.empty((len(rows), self.bpv), dtype=np.uint8)
        for fi in np.unique(f):
            k = np.flatnonzero(f == fi)
            out[k] = _pread_rows(self._fds[fi], self._data_off[fi], self.bpv, rows[k] - self._starts[fi],
                                 max_gap=_COALESCE_GAP, path=self.files[fi])
        return out

    def __del__(self):
        for fd in getattr(self, "_fds", []):
            try:
                os.close(fd)
            except OSError:
                pass

    def stored_stats(self, rows):
        return self._mu[rows], self._sxx[rows], self._maf[rows]

    def variant_ids(self, rows=None):
        if self._ann is None or "ID" not in self._ann.column_names:
            return None
        c = self._ann.column("ID")
        return c if rows is None else c.take(rows)

    def alleles(self, rows=None):
        if self._ann is None or "ALT" not in self._ann.column_names:
            return None, None
        cnt, oth = self._ann.column("ALT"), self._ann.column("REF")
        if rows is not None:
            cnt, oth = cnt.take(rows), oth.take(rows)
        return cnt, oth


def open_genotypes(path, **kw):
    """:class:`BedGenotypes` for a PLINK prefix (``x`` with ``x.bed``) or
    :class:`CugenGenotypes` for a directory of ``chr{c}.cugen`` files."""
    p = str(path)
    if p.endswith(".bed"):
        p = p[:-4]
    if os.path.exists(p + ".bed"):
        return BedGenotypes(p, **kw)
    if os.path.isdir(p):
        return CugenGenotypes(p, **kw)
    raise FileNotFoundError(f"{path}: neither a PLINK prefix nor a cugen directory")
