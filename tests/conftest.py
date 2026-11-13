"""Fixtures for the cugen test suite.

Everything here is CPU-only and network-free: .cugen files are synthesised in
tmp_path via cugen.write, which needs no GPU.
"""
import numpy as np
import pytest

from cugen.write import write_cugen

from cugen import cuda as _cuda

# GPU tests need a USABLE device, not just an importable cupy: on a CPU node with
# cupy installed `import cupy` succeeds and every CUDA call then fails, so an
# import probe turned skips into failures there. cugen.cuda.is_available()
# imports cupy, counts devices and runs a 1-element op (cached).
_HAS_CUPY = _cuda.is_available()
_NO_GPU_REASON = "CUDA not available: %s" % (_cuda.unavailable_reason(),)

requires_gpu = pytest.mark.skipif(not _HAS_CUPY, reason=_NO_GPU_REASON)


def pytest_collection_modifyitems(config, items):
    """Skip (not fail) every ``@pytest.mark.gpu`` test when CUDA is not usable."""
    if _HAS_CUPY:
        return
    skip = pytest.mark.skip(reason=_NO_GPU_REASON)
    for item in items:
        if "gpu" in item.keywords:
            item.add_marker(skip)

try:
    import cudf  # noqa: F401
    _HAS_CUDF = True
except ImportError:
    _HAS_CUDF = False

# The fused epilogue is only selected when cuDF is present (ld_matrix gates
# `on_device` on it), so a test that ASSERTS the fused path ran cannot run
# without cuDF. Skipping is the honest outcome there -- failing would report a
# missing optional dependency as a code defect.
requires_cudf = pytest.mark.skipif(
    not (_HAS_CUPY and _HAS_CUDF), reason="CuPy+cuDF required for fused path")


def simulate_haplotypes(n_samples, n_variants, seed=0, missing_rate=0.0,
                        straddle_half=True):
    """Genotypes with a spread of LD and MAF.

    straddle_half puts ALT frequency above 0.5 for half the variants. Without
    that the 'alt' and 'major' sign conventions coincide and any orientation
    test passes vacuously -- a real defect we hit while building this.
    """
    rng = np.random.default_rng(seed)
    latent = rng.random(n_samples)
    dos = np.zeros((n_variants, n_samples), dtype=np.uint8)
    for v in range(n_variants):
        mix = rng.uniform(0.0, 0.95)
        lat = mix * latent + (1 - mix) * rng.random(n_samples)
        if straddle_half and v % 2 == 0:
            af = rng.uniform(0.55, 0.92)
        else:
            af = rng.uniform(0.08, 0.45)
        t1, t2 = np.quantile(lat, [(1 - af) ** 2, 1 - af ** 2])
        dos[v] = (lat > t1).astype(np.uint8) + (lat > t2).astype(np.uint8)
    if missing_rate > 0:
        mask = rng.random((n_variants, n_samples)) < missing_rate
        dos[mask] = 3
    return dos


@pytest.fixture
def dosages():
    """(n_variants, n_samples) uint8, 3 = missing."""
    return simulate_haplotypes(200, 12, seed=11, missing_rate=0.0)


@pytest.fixture
def dosages_missing():
    d = simulate_haplotypes(200, 12, seed=12, missing_rate=0.0)
    rng = np.random.default_rng(5)
    d[3, rng.choice(200, 30, replace=False)] = 3
    d[7, rng.choice(200, 45, replace=False)] = 3
    return d


def _write(tmp_path, dos, name="chr22.cugen"):
    # write_cugen takes (n_samples, n_variants) -- transpose in.
    p = tmp_path / name
    write_cugen(str(p), dos.T.astype(np.uint8))
    return str(p)


@pytest.fixture
def small_cugen(tmp_path, dosages):
    return _write(tmp_path, dosages), dosages


@pytest.fixture
def missing_cugen(tmp_path, dosages_missing):
    return _write(tmp_path, dosages_missing), dosages_missing


@pytest.fixture
def write_cugen_file(tmp_path):
    def _f(dos, name="chr22.cugen"):
        return _write(tmp_path, dos, name)
    return _f


# ---------------------------------------------------------------------------
# Phased fixtures for cugen.impute
#
# Imputation needs haplotypes with real linkage. A panel of independent
# Bernoulli draws carries no information to transfer from typed to untyped
# markers, so an imputation test built on one measures nothing: every method,
# correct or broken, returns the allele frequency. These simulate a mosaic of
# founder haplotypes instead, which is the structure the Li-Stephens model is
# a model OF.
# ---------------------------------------------------------------------------

def simulate_mosaic(n_haplotypes, n_variants, n_founders=8, seed=0,
                    min_seg=80, max_seg=400, founder_freq=0.35):
    """Haplotypes as mosaics of `n_founders` founders. Returns (H, M) uint8."""
    rng = np.random.default_rng(seed)
    founders = (rng.random((n_founders, n_variants)) < founder_freq).astype(np.uint8)
    out = np.empty((n_haplotypes, n_variants), dtype=np.uint8)
    for i in range(n_haplotypes):
        who = int(rng.integers(0, n_founders))
        j = 0
        while j < n_variants:
            seg = int(rng.integers(min_seg, max_seg))
            out[i, j:j + seg] = founders[who, j:j + seg]
            who = int(rng.integers(0, n_founders))
            j += seg
    return out


def scaled_ne(n_ref_hap, base_ne=100_000, base_hap=4904):
    """Beagle's ne=100000 rescaled to a panel of `n_ref_hap` haplotypes.

    tau = 1 - exp(-4*ne*d/|H|) scales inversely with panel size, so the stock
    default drives tau towards 1 on the small panels tests can afford, and the
    HMM becomes memoryless. A test run that way passes (nothing raises) while
    being unable to distinguish a correct implementation from a broken one.
    """
    return base_ne * n_ref_hap / base_hap


@pytest.fixture
def phased_panel():
    """(ref_hap, tgt_hap_full, typed_idx, pos) for a small imputation problem."""
    n_ref_hap, n_tgt_hap, n_var = 400, 40, 2000
    hap = simulate_mosaic(n_ref_hap + n_tgt_hap, n_var, seed=17)
    rng = np.random.default_rng(17)
    pos = np.sort(rng.choice(np.arange(1, 30_000_000), size=n_var, replace=False))
    typed = np.sort(rng.choice(n_var, size=n_var // 10, replace=False))
    return hap[:n_ref_hap], hap[n_ref_hap:], typed, pos


@pytest.fixture
def phased_cugen(tmp_path, phased_panel):
    """Write the phased panel to disk; returns (ref_path, tgt_path, meta)."""
    import pandas as pd
    from cugen.write import write_cugen_phased
    ref_hap, tgt_hap, typed, pos = phased_panel
    ref_p = tmp_path / "ref.cugen"
    tgt_p = tmp_path / "tgt.cugen"
    write_cugen_phased(str(ref_p), ref_hap)
    write_cugen_phased(str(tgt_p), tgt_hap[:, typed], gidx=typed)
    ann = pd.DataFrame({"gidx": np.arange(pos.size), "POS": pos,
                        "CHR": "20", "ID": [f"v{i}" for i in range(pos.size)]})
    return str(ref_p), str(tgt_p), {"ann": ann, "typed": typed,
                                    "ref_hap": ref_hap, "tgt_hap": tgt_hap,
                                    "ne": scaled_ne(ref_hap.shape[0])}


# ---------------------------------------------------------------------------
# Small cohorts for cugen.genotypes / cugen.background: a polygenic phenotype on
# three chromosomes, written as a PLINK .bed (counting A1) and as per-chromosome
# cugens (counting A2, as bed2cugen does by default).
# ---------------------------------------------------------------------------

def simulate_cohort(n=300, m_per_chr=(60, 50, 40), h2=0.5, n_causal=20, miss=0.02, seed=0, ncov=2):
    """Counts (variants, samples) with 3 = missing, chrom (variants,), y (n,), covariates (n, ncov)."""
    rng = np.random.default_rng(seed)
    chrom = np.concatenate([np.full(m, c + 1) for c, m in enumerate(m_per_chr)])
    m = len(chrom)
    p = rng.uniform(0.05, 0.5, m)
    G = rng.binomial(2, p[:, None], size=(m, n)).astype(np.uint8)
    G[rng.random((m, n)) < miss] = 3
    X = np.where(G == 3, 0, G).astype(float)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    beta = np.zeros(m)
    cs = rng.choice(m, n_causal, replace=False)
    beta[cs] = rng.normal(0, 1, n_causal)
    gv = beta @ X
    gv = gv / gv.std() * np.sqrt(h2)
    C = rng.normal(size=(n, ncov))
    y = gv + C @ rng.normal(0, 0.3, ncov) + rng.normal(0, np.sqrt(1 - h2), n)
    return G, chrom, y, C


def write_bed(prefix, counts_vn, chrom, pos=None, ids=None, a1=None, a2=None, fam_ids=None):
    """Counts of A1 (variants, samples), 3 = missing -> prefix.bed/.bim/.fam (PLINK 1, SNP-major)."""
    from cugen._codec import BED_MAGIC, pack_bed
    counts_vn = np.asarray(counts_vn)
    m, n = counts_vn.shape
    with open(prefix + ".bed", "wb") as fh:
        fh.write(BED_MAGIC)
        fh.write(pack_bed(counts_vn, counted="A1").tobytes())
    pos = np.arange(1, m + 1) * 1000 if pos is None else pos
    ids = [f"snp{i}" for i in range(m)] if ids is None else ids
    a1 = ["A"] * m if a1 is None else a1
    a2 = ["G"] * m if a2 is None else a2
    with open(prefix + ".bim", "w") as fh:
        for i in range(m):
            fh.write(f"{chrom[i]}\t{ids[i]}\t0\t{pos[i]}\t{a1[i]}\t{a2[i]}\n")
    fam_ids = [f"s{i}" for i in range(n)] if fam_ids is None else fam_ids
    with open(prefix + ".fam", "w") as fh:
        for s in fam_ids:
            fh.write(f"{s} {s} 0 0 0 -9\n")
    return prefix


def write_counts_cugen(path, counts_vn, gidx=None):
    """Counts (variants, samples), 3 = missing -> a 2-bit cugen at `path`."""
    g = np.asarray(counts_vn, dtype=np.float64).T.copy()
    g[g == 3] = np.nan
    return write_cugen(str(path), g, gidx=gidx)


def a2_counts(G):
    """A1 counts -> A2 counts (missing stays 3)."""
    return np.where(G == 3, 3, 2 - G).astype(np.uint8)


def make_cohort_files(d, n=240, m_per_chr=(70, 55, 45), seed=3):
    """Write the simulated cohort under directory `d`; returns a dict of arrays and readers."""
    from cugen.genotypes import BedGenotypes, CugenGenotypes
    d = str(d)
    G, chrom, y, C = simulate_cohort(n=n, m_per_chr=m_per_chr, seed=seed)
    write_bed(d + "/t", G, chrom)
    for c in np.unique(chrom):
        o = chrom == c
        write_counts_cugen(f"{d}/chr{c}.cugen", a2_counts(G[o]), gidx=np.flatnonzero(o))
    return dict(dir=d, G=G, chrom=chrom, y=y, C=C, bed=BedGenotypes(d + "/t"), cg=CugenGenotypes(d))


@pytest.fixture(scope="module")
def cohort(tmp_path_factory):
    """Simulated cohort (n=240, 170 variants on 3 chromosomes) as .bed and cugens."""
    return make_cohort_files(tmp_path_factory.mktemp("cohort"))
