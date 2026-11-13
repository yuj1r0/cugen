"""Tests for cugen.datasets.simulate_height and the evaluators (tiny sizes, CPU only)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from cugen._codec import BED_MAGIC, pack_bed, unpack_bed, unpack_cugen
from cugen.datasets import (PRESETS, clump_loci, evaluate_gwas, simulate_height, verify,
                            workflow_config)
from cugen.write import validate_cugen, variant_stats

SMALL = dict(n=600, m=3000, n_chrom=3, n_causal=60, n_large=3, block_mean=50,
             n_test=150, h2=0.6, seed=7)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _cugen_dosages(path):
    """(n_samples, n_variants) float32 dosages of a whole 2-bit cugen, NaN = missing."""
    from cugen.io import CugenReader
    r = CugenReader(str(path))
    try:
        raw = np.frombuffer(r.read_packed_bytes(0, r.n_variants), dtype=np.uint8)
        codes = unpack_cugen(raw.reshape(r.n_variants, r.bytes_per_variant), r.n_samples)
    finally:
        r.close()
    x = codes.T.astype(np.float32)
    x[codes.T == 3] = np.nan
    return x


def _bed_counts(bed, n):
    """(n_variants, n) bim-A1 counts of a PLINK1 .bed, 3 = missing."""
    raw = Path(bed).read_bytes()
    assert raw[:3] == BED_MAGIC
    bpv = (n + 3) // 4
    return unpack_bed(np.frombuffer(raw[3:], dtype=np.uint8).reshape(-1, bpv), n, counted="A1")


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    d = tmp_path_factory.mktemp("height")
    meta = simulate_height(d, verbose=False, **SMALL)
    return d, meta


def _all_dosages(d, n_chrom, sub=""):
    return np.concatenate([_cugen_dosages(Path(d) / sub / "cugen" / f"chr{c}.cugen")
                           for c in range(1, n_chrom + 1)], axis=1)          # (n, m)


def test_files_and_meta(ds):
    d, meta = ds
    for f in ("pheno.tsv", "covar.tsv", "truth.tsv", "annotation.feather", "meta.json",
              "README.md", "samples.psam", "cohort_ref.npz", "config_ultralasso.json",
              "ld_blocks.tsv", "bed/height.bed", "bed/height.bim", "bed/height.fam",
              "test/pheno.tsv", "test/bed/height_test.bed", "test/cugen/chr1.cugen"):
        assert (d / f).exists(), f
    assert not list((d / "bed").glob(".*bedpart"))
    assert sum(meta["n_variants_per_chrom"].values()) == SMALL["m"]
    assert meta["numpy_version"] == np.__version__
    assert meta["params"]["chrom_lengths"] == "grch37"
    assert meta["allele_coding"]["counted_allele"] == "ALT"
    ph = pd.read_csv(d / "pheno.tsv", sep="\t")
    assert list(ph.columns[:5]) == ["FID", "IID", "height", "age", "sex"]
    assert len(ph) == SMALL["n"]
    assert abs(ph["height"].mean()) < 1e-4 and abs(ph["height"].std(ddof=0) - 1) < 1e-4
    ann = pd.read_feather(d / "annotation.feather")
    assert list(ann.columns) == ["gidx", "CHR", "POS", "ID", "REF", "ALT"]
    assert (ann["gidx"].to_numpy() == np.arange(SMALL["m"])).all()
    assert (ann.groupby("CHR")["POS"].apply(lambda s: s.is_monotonic_increasing)).all()
    assert (ann["REF"] != ann["ALT"]).all()
    truth = pd.read_csv(d / "truth.tsv", sep="\t")
    assert len(truth) == SMALL["n_causal"] and truth["is_large"].sum() == SMALL["n_large"]


def test_meta_checksums_and_verify(ds, tmp_path):
    d, meta = ds
    files = meta["files"]
    assert "meta.json" not in files and "README.md" in files and "bed/height.bed" in files
    for rel, rec in files.items():
        assert rec["sha256"] == _sha(d / rel) and rec["size"] == (d / rel).stat().st_size, rel
    rep = verify(d, "demo_height_n3k")
    assert rep["ok"] and rep["reference"] == "meta.json"
    assert set(rep["contents"]) == set(files) and set(rep["contents"].values()) == {"ok"}
    # a modified copy is reported, file by file
    d2 = tmp_path / "copy"
    import shutil
    shutil.copytree(d, d2)
    p = d2 / "truth.tsv"
    p.write_bytes(p.read_bytes().replace(b"\t", b" ", 1))
    rep2 = verify(d2, "demo_height_n3k")
    assert not rep2["ok"] and rep2["contents"]["truth.tsv"] == "sha256_mismatch"
    assert rep2["contents"]["pheno.tsv"] == "ok"


def test_regenerate_command(ds):
    d, meta = ds
    cmd = meta["regenerate"]
    assert cmd.startswith("python -m cugen.datasets.simulate_height --out DIR")
    for frag in ("--n 600", "--m 3000", "--n-chrom 3", "--seed 7", "--n-test 150", "--h2 0.6"):
        assert frag in cmd, (frag, cmd)
    assert "--preset" not in cmd and "--chrom-lengths" not in cmd
    assert cmd in (d / "README.md").read_text()
    from cugen.datasets.simulate_height import _parser
    a = _parser().parse_args(cmd.split()[3:])          # the command parses with the CLI
    assert (a.n, a.m, a.n_chrom, a.seed, a.n_test) == (600, 3000, 3, 7, 150)


def test_determinism(ds, tmp_path):
    d, _ = ds
    d2 = tmp_path / "again"
    simulate_height(d2, n_workers=2, verbose=False, checksums=False, **SMALL)  # same seed, parallel
    for f in ["cugen/chr1.cugen", "cugen/chr3.cugen", "bed/height.bed", "bed/height.bim",
              "pheno.tsv", "truth.tsv", "test/cugen/chr2.cugen", "ld_blocks.tsv",
              "config_ultralasso.json"]:
        assert _sha(d / f) == _sha(d2 / f), f
    pd.testing.assert_frame_equal(pd.read_feather(d / "annotation.feather"),
                                  pd.read_feather(d2 / "annotation.feather"))
    d3 = tmp_path / "other"
    simulate_height(d3, verbose=False, checksums=False, **{**SMALL, "seed": 8, "formats": ("cugen",)})
    assert _sha(d / "cugen/chr1.cugen") != _sha(d3 / "cugen/chr1.cugen")
    assert not (d3 / "bed").exists()


def test_presets_and_parameter_checks(tmp_path):
    assert set(PRESETS) == {"n20k", "n3k"}
    assert PRESETS["n3k"] == dict(n=3_000, n_test=1_000, m=500_000)
    with pytest.raises(ValueError, match="unknown preset"):
        simulate_height(tmp_path / "x", preset="n1m")
    with pytest.raises(TypeError, match="unknown parameters"):
        simulate_height(tmp_path / "x", n_samples=10)
    # a preset with overrides: the overrides win, the rest comes from the preset
    meta = simulate_height(tmp_path / "p", preset="n3k", verbose=False, checksums=False,
                           **{**SMALL, "n": 400, "m": 1500, "n_test": 0, "formats": ("cugen",)})
    assert meta["preset"] == "n3k" and meta["params"]["n"] == 400 and meta["params"]["n_test"] == 0
    assert "--preset n3k" in meta["regenerate"]


def test_equal_length_chromosomes_beyond_22(tmp_path):
    meta = simulate_height(tmp_path / "c25", verbose=False, checksums=False,
                           **{**SMALL, "n": 300, "n_test": 0, "m": 2500, "n_chrom": 25,
                              "n_large": 0, "n_causal": 25, "formats": ("cugen",)})
    assert meta["params"]["chrom_lengths"] == "equal"
    per = np.array([meta["n_variants_per_chrom"][str(c)] for c in range(1, 26)])
    assert per.sum() == 2500 and per.max() - per.min() <= 1
    assert not (tmp_path / "c25" / "config_ultralasso.json").exists()   # workflow: chr 1..22 only
    with pytest.raises(ValueError, match="grch37"):
        simulate_height(tmp_path / "bad", verbose=False, **{**SMALL, "n_chrom": 25, "chrom_lengths": "grch37"})


def test_h2_and_variance_shares(ds):
    d, meta = ds
    r = meta["realised"]
    assert abs(r["h2"] - SMALL["h2"]) < 0.05
    assert abs(r["var_g"] - SMALL["h2"]) < 1e-9          # exact scaling in training
    truth = pd.read_csv(d / "truth.tsv", sep="\t")
    # marginal shares ignore LD between causal variants: allow slack
    assert abs(truth["var_explained"].sum() - SMALL["h2"]) < 0.15
    assert abs(r["large_var_explained"] - 0.10 * SMALL["h2"]) < 0.04
    # the true genetic value explains ~h2 of the covariate-adjusted trait
    gv = np.load(d / "genetic_value.npz")
    ph = pd.read_csv(d / "pheno.tsv", sep="\t")
    X = np.column_stack([np.ones(len(ph)), ph[["age", "sex"]].to_numpy(float)])
    y = ph["height"].to_numpy(float)
    yr = y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
    g = gv["g"][gv["split"] == 0]
    assert abs(np.corrcoef(g, yr)[0, 1] ** 2 - SMALL["h2"]) < 0.08
    # truth betas are per ALT allele on the standardised height scale
    G = _all_dosages(d, SMALL["n_chrom"])
    gx = (G[:, truth["gidx"].to_numpy()] - G[:, truth["gidx"].to_numpy()].mean(0)) @ truth["beta"].to_numpy()
    assert np.corrcoef(gx, g)[0, 1] > 0.9999


def test_ld_decays(ds):
    d, _ = ds
    G = _all_dosages(d, SMALL["n_chrom"]).astype(np.float64)
    Z = (G - G.mean(0)) / G.std(0)
    blocks = pd.read_csv(d / "ld_blocks.tsv", sep="\t")
    bid = np.repeat(np.arange(len(blocks)), blocks["n_variants"].to_numpy())
    n = Z.shape[0]

    def r2(i, j):
        return (np.einsum("ij,ij->j", Z[:, i], Z[:, j]) / n) ** 2

    i = np.arange(Z.shape[1] - 20)
    same1 = bid[i] == bid[i + 1]
    same20 = bid[i] == bid[i + 20]
    r_lag1 = r2(i[same1], i[same1] + 1).mean()
    r_lag20 = r2(i[same20], i[same20] + 20).mean()
    rng = np.random.default_rng(0)
    a, b = rng.integers(0, Z.shape[1], (2, 20000))
    diff = bid[a] != bid[b]
    r_between = r2(a[diff], b[diff]).mean()
    assert r_lag1 > 0.2                         # strong LD between neighbours
    assert r_lag1 > 1.5 * r_lag20               # decays inside a block
    assert r_lag20 > 3 * r_between              # still above the between-block level
    assert r_between < 3.0 / n                  # ~ 1/n: no LD between blocks


def test_cugen_readback(ds):
    d, meta = ds
    import cugen.io as cio
    from cugen.datasets.evaluate import _read_cugen_rows
    gidx0 = 0
    for c in range(1, SMALL["n_chrom"] + 1):
        p = str(d / "cugen" / f"chr{c}.cugen")
        assert validate_cugen(p, verbose=False)["ok"]
        r = cio.CugenReader(p)
        mc = meta["n_variants_per_chrom"][str(c)]
        assert (r.n_samples, r.n_variants) == (SMALL["n"], mc)
        assert r.encoding == 0 and r.has_gidx_map and not r.has_missing
        assert (r.gidx == np.arange(gidx0, gidx0 + mc)).all()
        X = _cugen_dosages(p)
        assert np.array_equal(r.read_to_numpy(0, 5), X[:, :5])     # package reader agrees
        rows = np.array([mc - 1, 3, 0, 17])
        assert np.array_equal(_read_cugen_rows(p, rows), X[:, rows])  # random-access helper agrees
        st = np.array([variant_stats(X[:, j])[:3] for j in range(mc)])
        mu, sxx, maf = r.get_stats()
        np.testing.assert_allclose(mu, st[:, 0], rtol=1e-6)
        np.testing.assert_allclose(sxx, st[:, 1], rtol=1e-5)
        np.testing.assert_allclose(maf, st[:, 2], rtol=1e-6)
        assert maf.min() >= SMALL.get("maf_min", 0.01) - 1e-9
        r.close()
        gidx0 += mc
    assert (Path(p + ".samples.txt").read_text().split()[0]) == "1000001"


def _pack_bed_reference(G):
    """PLINK1 packing written out by hand: ALT dosage 0/1/2 -> 11/10/00, sample 0 in the LOW bits."""
    lut = np.array([3, 2, 0], dtype=np.uint8)
    L, n = G.shape
    c = lut[G]
    pad = (-n) % 4
    if pad:
        c = np.concatenate([c, np.zeros((L, pad), dtype=np.uint8)], axis=1)
    c = c.reshape(L, -1, 4)
    return (c[:, :, 0] | (c[:, :, 1] << 2) | (c[:, :, 2] << 4) | (c[:, :, 3] << 6)).astype(np.uint8)


@pytest.mark.parametrize("n", [1, 4, 7, 600])
def test_bed_packing_matches_plink_reference(n):
    G = np.random.default_rng(n).integers(0, 3, size=(13, n)).astype(np.int8)
    ref = _pack_bed_reference(G)
    assert np.array_equal(pack_bed(G, counted="A1"), ref)
    assert np.array_equal(unpack_bed(ref, n, counted="A1"), G.astype(np.uint8))
    assert np.array_equal(unpack_bed(ref, n, counted="A2"), (2 - G).astype(np.uint8))


def test_bed_matches_cugen(ds):
    d, _ = ds
    G = _all_dosages(d, SMALL["n_chrom"])
    B = _bed_counts(d / "bed" / "height.bed", SMALL["n"]).T
    assert B.shape == G.shape and np.array_equal(B.astype(np.float32), G)
    Gt = _all_dosages(d, SMALL["n_chrom"], sub="test")
    Bt = _bed_counts(d / "test" / "bed" / "height_test.bed", SMALL["n_test"]).T
    assert np.array_equal(Bt.astype(np.float32), Gt)
    bim = pd.read_csv(d / "bed" / "height.bim", sep="\t", header=None)
    ann = pd.read_feather(d / "annotation.feather")
    assert (bim[1].to_numpy() == ann["ID"].to_numpy()).all()
    assert (bim[4].to_numpy() == ann["ALT"].to_numpy()).all()      # A1 = ALT (counted)
    assert (bim[5].to_numpy() == ann["REF"].to_numpy()).all()
    fam = pd.read_csv(d / "bed" / "height.fam", sep="\t", header=None)
    ph = pd.read_csv(d / "pheno.tsv", sep="\t")
    assert (fam[0].to_numpy() == ph["FID"].to_numpy()).all()
    assert ((fam[4] == 1).to_numpy() == (ph["sex"] == 1).to_numpy()).all()


def _hudson(G, pop, P):
    vals = []
    for a in range(P):
        for b in range(a + 1, P):
            xa, xb = G[pop == a], G[pop == b]
            na, nb = 2 * len(xa), 2 * len(xb)
            pa, pb = xa.mean(0) / 2, xb.mean(0) / 2
            num = (pa - pb) ** 2 - pa * (1 - pa) / (na - 1) - pb * (1 - pb) / (nb - 1)
            den = pa * (1 - pb) + pb * (1 - pa)
            vals.append(num.sum() / den.sum())
    return float(np.mean(vals))


def test_structure_fst(tmp_path):
    kw = {**SMALL, "n": 1500, "n_test": 0, "m": 4000, "n_pops": 3, "fst": 0.05,
          "strat_shift": 0.5, "formats": ("cugen",)}
    d = tmp_path / "pops"
    meta = simulate_height(d, verbose=False, checksums=False, **kw)
    ph = pd.read_csv(d / "pheno.tsv", sep="\t")
    assert "pop" in ph.columns and sorted(ph["pop"].unique()) == [0, 1, 2]
    G = _all_dosages(d, kw["n_chrom"]).astype(np.float64)
    fst = _hudson(G, ph["pop"].to_numpy(), 3)
    assert 0.03 < fst < 0.07, fst
    # PCs separate the populations; strat_shift moves the population means
    assert ph.groupby("pop")["PC1"].mean().abs().max() > 3 * ph["PC1"].std() / np.sqrt(500)
    m_ = ph.groupby("pop")["height"].mean()
    assert m_[2] - m_[0] > 0.3
    assert meta["realised"]["var_popshift"] > 0
    # homogeneous default: Fst ~ 0
    d0 = tmp_path / "homog"
    simulate_height(d0, verbose=False, checksums=False, **{**kw, "n_pops": 1, "fst": 0.0, "strat_shift": 0.0})
    G0 = _all_dosages(d0, kw["n_chrom"]).astype(np.float64)
    assert abs(_hudson(G0, ph["pop"].to_numpy(), 3)) < 0.005


def test_workflow_config(ds):
    d, _ = ds
    from cugen.config import load_and_validate_config
    cfg = load_and_validate_config(workflow_config(d, d / "res"))
    for k in ("cugen_dir", "annotation", "phe_file"):
        assert Path(cfg["data"][k]).exists()
    assert cfg["data"]["chromosomes"] == [1, 2, 3]
    assert cfg["covariates"]["columns"][:2] == ["age", "sex"]
    assert cfg["gpu"]["n_workers_gwas"] == 1 and cfg["gpu"]["cuda_mps"] is False   # small n
    big = workflow_config(d, d / "res", n=400_000)
    assert "gpu" not in big                                                  # package default
    assert workflow_config(d, d / "res", chromosomes=[2, 1])["data"]["chromosomes"] == [1, 2]
    rel = json.load(open(d / "config_ultralasso.json"))
    assert rel["data"]["cugen_dir"] == "cugen" and rel["cohort"]["n_samples"] == SMALL["n"]


def test_workflow_config_refuses_unsupported_options(ds):
    d, _ = ds
    # the core config of this release has no step-3 alpha_scaling key: refuse rather than
    # write a key the workflow would silently ignore
    with pytest.raises(NotImplementedError, match="alpha_scaling"):
        workflow_config(d, d / "res", alpha_scaling="sqrt_n")
    with pytest.raises(ValueError, match="alpha_scaling"):
        workflow_config(d, d / "res", alpha_scaling="bogus")
    for bg in ("cugenie", "cugenie_k", "ultraloco"):
        with pytest.raises(NotImplementedError, match="background"):
            workflow_config(d, d / "res", background=bg)
    with pytest.raises(ValueError, match="chromosomes"):
        workflow_config(d, d / "res", chromosomes=range(1, 39))


def _clump_reference(df, p_threshold=5e-8, clump_kb=500):
    """The original greedy loop: GWS variants by P; a new lead unless an earlier lead is within clump_kb."""
    g = df[df["P"] < p_threshold].sort_values("P", kind="stable")
    rows = []
    w = clump_kb * 1000
    for chrom, sub in g.groupby("CHR", sort=True):
        pos = sub["POS"].to_numpy()
        leads, counts = [], []
        for k, ps in enumerate(pos):
            hit = next((li for li, lp in enumerate(leads) if abs(ps - pos[lp]) <= w), None)
            if hit is None:
                leads.append(k)
                counts.append(1)
            else:
                counts[hit] += 1
        for lp, cnt in zip(leads, counts):
            r = sub.iloc[lp]
            rows.append(dict(CHR=int(chrom), POS=int(r["POS"]), P=float(r["P"]), ID=str(r["ID"]),
                             gidx=int(r["gidx"]), n_gws=int(cnt)))
    return pd.DataFrame(rows, columns=["CHR", "POS", "P", "ID", "gidx", "n_gws"])


def test_clump_loci_matches_reference_loop():
    rng = np.random.default_rng(3)
    m = 4000
    chrom = np.sort(rng.integers(1, 4, m))
    pos = np.concatenate([np.sort(rng.choice(np.arange(1, 20_000_000, 50), (chrom == c).sum(), replace=False))
                          for c in (1, 2, 3)])
    p = 10.0 ** -rng.uniform(0, 14, m)
    p[rng.choice(m, 40, replace=False)] = 0.0           # underflowed P values tie
    p[rng.choice(m, 40, replace=False)] = 1e-9          # and so do repeated values
    df = pd.DataFrame({"CHR": chrom, "POS": pos, "ID": [f"v{i}" for i in range(m)], "gidx": np.arange(m), "P": p})
    for kb in (100, 500):
        new = clump_loci(df, clump_kb=kb)
        ref = _clump_reference(df, clump_kb=kb)
        pd.testing.assert_frame_equal(new.reset_index(drop=True), ref, check_dtype=False)


def test_evaluate_marginal_scan(ds):
    d, _ = ds
    G = _all_dosages(d, SMALL["n_chrom"]).astype(np.float64)
    ph = pd.read_csv(d / "pheno.tsv", sep="\t")
    y = ph["height"].to_numpy(float)
    y = y - y.mean()
    Gc = G - G.mean(0)
    b = Gc.T @ y / (Gc ** 2).sum(0)
    res = y[:, None] - Gc * b
    se = np.sqrt((res ** 2).sum(0) / (len(y) - 2) / (Gc ** 2).sum(0))
    from scipy.stats import norm
    from cugen.loci import lambda_gc
    ann = pd.read_feather(d / "annotation.feather")
    ss = ann.assign(BETA=b, SE=se, Z=b / se, P=2 * norm.sf(np.abs(b / se)))
    r = evaluate_gwas(ss, d)
    assert r["n_tested"] == SMALL["m"] and r["n_large"] == SMALL["n_large"]
    assert 0 <= r["n_false_loci"] <= r["n_loci"]
    assert np.isfinite(r["lambda_gc"]) and r["n_loci"] >= 1
    assert r["lambda_gc"] == lambda_gc(z=ss["Z"].to_numpy())             # one lambda definition
    # from P alone (no Z column) the same genomic control is recovered
    r_p = evaluate_gwas(ss.drop(columns="Z"), d)
    assert abs(r_p["lambda_gc"] - r["lambda_gc"]) < 1e-3 and r_p["n_loci"] == r["n_loci"]
    # a perfect "oracle" sumstats recovers every causal variant with no false loci
    truth = pd.read_csv(d / "truth.tsv", sep="\t")
    oracle = ann.assign(P=1.0, Z=0.0)
    oracle.loc[truth["gidx"], "P"] = 1e-12
    oracle.loc[truth.loc[truth["is_large"] == 1, "gidx"], "P"] = 1e-30   # large loci lead first
    o = evaluate_gwas(oracle, d)
    assert o["n_large_recovered"] == SMALL["n_large"] and o["n_false_loci"] == 0


def test_cli_small(tmp_path, capsys):
    from cugen.datasets.simulate_height import main
    main(["--out", str(tmp_path / "cli"), "--n", "200", "--m", "600", "--n-chrom", "2", "--n-causal", "10",
          "--n-large", "1", "--block-mean", "40", "--n-test", "0", "--formats", "cugen", "--quiet",
          "--no-checksums"])
    out = json.loads(capsys.readouterr().out)
    assert set(out) == {"realised", "timings_s", "sizes_bytes"}
    meta = json.load(open(tmp_path / "cli" / "meta.json"))
    assert meta["params"]["n"] == 200 and meta["params"]["formats"] == ["cugen"] and "files" not in meta
