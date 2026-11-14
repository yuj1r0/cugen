<p align="center">
  <img src="cugen.png" width="560" alt="CuGen logo" />
</p>

<h3 align="center">A GPU-accelerated framework for large-scale genomics.</h3>

---

## What CuGen is

CuGen is the world's first fully GPU-native framework designed for biobank-scale genomics. Its GPU-acceleration architecture centers around three core innovations:

1. **`ultraLasso`** — a memory and computationally efficient sparse 
   hierarchical regression framework built
   upon uniLasso, producing compact phenotype-informed predictors
   optimized for confounding control through leave-one-chromosome-out
   (LOCO) predictions. The same active set powers both downstream GWAS
   and fine-mapping (UltraSuSie) in a single pipeline as embarrassingly
   parallelizable operations. 

2. **The `.cugen` file format** — *Compute Unified Genomes*. A
   CUDA-optimized 2-bit-packed variant-major genotype container with
   pre-computed per-variant statistics, fixed-size records for O(1)
   random access, and a per-variant global index for joining with
   annotations and imputed datasets. The same file supports both
   *streaming* mode (pinned host + async H2D for ultraLasso's sequential
   scans) and *random-access* mode (direct `pread` for UltraSuSiE's
   per-locus sufficient statistics). Fine-mapping leverages in-sample 
   LD via `.cugen` random-access, avoiding external reference-panel mismatches.

3. **An integrated GPU-native toolkit** for routine genomic analyses,
   including QC, allele-frequency computation, polygenic scoring, and
   visualization — simplifying workflows by removing unnecessary file
   conversions and software juggling. In beta version the toolkit is not
   yet complete, but the goal is to perform all operations eventually on GPUs. 

Using CuGen starts by converting your genomes to `.cugen` file format. 

Together these enable a full GWAS-pipeline, including fine-mapping, to be run in UKBB in as less as 10 minutes. 

The full ultraLasso pipeline lives under the `cg.ultralasso.*`
namespace. Each stage maps to one function call:

```
                  .cugen files (2-bit packed, variant-major, pre-computed stats)
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 1+2  cg.ultralasso.screen(...)                                │
  │  per-chr streaming univariate screen + local LASSO                  │
  └─────────────────────────────────────────────────────────────────────┘
                          │  candidate set per chromosome
                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 3    cg.ultralasso.fit(...)                                   │
  │  genome-wide joint LASSO + LOCO predictions                         │
  │  ➜ compact active set + LOCO residuals per chromosome               │
  └─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 4    cg.ultralasso.gwas(...)                                  │
  │  Wald / score test against LOCO residuals                           │
  │  (multi-process Pool(spawn) over CUDA MPS workers)                  │
  └─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────────────┐
  │  STEP 5    cg.ultralasso.loci(...)   +   cg.ultralasso.susie(...)   │
  │            cg.ultralasso.map_(...)                                  │
  │  Locus definition + UltraSuSiE (Tier 1) / UltraMAP (Tier 2)         │
  │  fine-mapping on EXACT in-sample LD                                 │
  └─────────────────────────────────────────────────────────────────────┘
                          │
                          ▼
                  per-locus credible sets + PIPs
```

End-to-end with one call: `cg.ultralasso.run(cohort_npz=..., cugen_dir=...)`.

## Install

```bash
pip install cugen
# or, with optional extras
pip install "cugen[gpu,plot]"
```

Requires Python ≥ 3.10. The biobank-scale UltraLasso pipeline needs CUDA 12.x and a recent
NVIDIA GPU (80 GB device memory recommended). The background engines in `cg.background`
(CuGenie, CuGenie-K, UltraLOCO) and the blocked step 4 also run on CPU with only numpy,
scipy, pandas and pyarrow; see [Running on CPU](#running-on-cpu).

## Quickstart

Drive the full pipeline from a single JSON config:

```python
import cugen as cg

cg.check_gpu()
cg.ultra_workflow("config.json")
```

Or stage-by-stage via the `cg.ultralasso.*` API:

```python
import cugen as cg
from cugen import ultralasso

# Build a residualised cohort NPZ (one-time per phenotype)
cg.prepare_cohort(
    phenotype="trait",
    ref_npz="ref_cohort.npz", master_phe="master.tsv", psam="data.psam",
    n_samples=N, covariates=["age", "sex"] + [f"PC{i}" for i in range(1, 11)],
    output="cohort.npz",
)

# Step 1+2 — per-chr screening + local LASSO
cands = ultralasso.screen(chrom=22, cohort_npz="cohort.npz",
                            cugen_path="data/chr22.cugen", alpha=6e-4)

# Step 3 — joint LASSO + LOCO residuals
fit = ultralasso.fit(cands, cohort_npz="cohort.npz", cugen_dir="data/",
                      covariates=COVARS, phe_file="master.tsv",
                      alpha=6e-3, ridge=1e-2, loco_mode="ols")

# Step 4 — GWAS (PLINK alias: cg.glm; Hail alias: cg.linear_regression_rows)
sumstats = ultralasso.gwas(cugen_dir="data/", cohort_npz="cohort.npz",
                            loco_predictions=fit["loco_predictions"],
                            n_workers=8)

# Step 5 — locus calling + in-sample fine-mapping (one stage, two calls)
loci = ultralasso.loci(sumstats, p_threshold=5e-8, window_kb=500)
fm   = ultralasso.susie(loci, fit["loco_predictions"], cugen_dir="data/",
                         annotation="annotations.feather",
                         output_dir="finemap/")
```

The `cg.ultralasso` namespace also exposes a top-level `run(...)` that
chains all five stages from a cohort NPZ.

Common operations (PLINK / plink2 / Hail equivalents in parentheses):

```python
cg.frequency("data/chr22.cugen")               # ≈ plink2 --freq
cg.variant_qc("data/", maf_min=1e-3)           # ≈ plink2 --maf / --geno / --hwe
cg.sample_qc("data/", missing_rate_max=0.02)   # ≈ plink2 --mind
cg.subset_cugen_dir("data/", "subset/", keep_samples=ids, n_workers=4)
                                                # ≈ plink2 --keep + write
cg.score(weights="weights.tsv", cugen_dir="data/")  # ≈ plink2 --score
cg.plot.qq(sumstats); cg.plot.manhattan(sumstats)
```

## Choosing a background

The *background* is the polygenic LOCO model that step 4 tests each variant against. Every
engine in `cg.background` writes the same `LocoPredictions` contract
(`loco_predictions.npz`: one prediction column per chromosome, a `chromosomes` field and the
sample IDs), so any of them can feed step 4.

| engine | call | method | use it for |
|---|---|---|---|
| UltraLasso | `cg.fit_joint_lasso` (steps 1–3, `cg.background.ultralasso`) | joint LASSO + LOCO OLS refits | biobank scale (n in the hundreds of thousands); needs CUDA |
| CuGenie | `cg.fit_cugenie` | REGENIE v2.2.4-exact step 1: block ridge, ridge stacking, masked LOCO | small and mid-size cohorts; CPU or GPU |
| CuGenie-K | `cg.fit_cugenie_k` | whole-genome kernel ridge, exact leave-one-out CV | n up to a few thousand (n × n kernels) |
| UltraLOCO | `cg.fit_ultraloco` + `cg.ultraloco_gwas` | GLS-LOCO mixed model (GCTA-MLMA-LOCO equivalent), its own whitened test | small cohorts with relatedness or structure |

Scope rule: at small n (a few thousand samples) the lasso background overfits (in-sample LOCO
R² near 1), the LOCO offset absorbs the signal and the Manhattan plot goes flat; use CuGenie or
UltraLOCO there, which are the calibrated small-n choices. CuGenie-K with the plain offset test
deflates a structured null (λ_GC ≈ 0.92 in our tests); `fit_ultraloco(scaling="regenie",
pairwise_count=False)` uses the same kernel for GLS whitening (the MLMA-whitened CuGenie-K).

Options worth knowing:
* `fit_cugenie(loco="masked" | "refit" | "both")`: masked is REGENIE-exact; refit refits level 1
  per chromosome.
* `fit_cugenie_k(offset="loo" | "insample", lam_grid="wide" | "stable" | array, null_tol=)`:
  `"wide"` (1e-7..1e3, default) reproduces the validated reference; below λ ≈ 1e-5 the LOO-CV
  choice is rounding-dominated (a warning says so) and `"stable"` (1e-5..1e3) gives
  implementation-stable choices.
* `fit_ultraloco(grm_snps=, pairwise_count=True, scaling="gcta" | "regenie")`,
  `ultraloco_gwas(reml_bounds=(1e-3, 1e4))`; `ultraloco_gwas(geno, y, covariates, ...)` also
  runs end to end.
* `chromosomes=` on every call restricts the analysis; the LOCO columns are exactly those
  chromosomes (default: 1..max chromosome of the panel, so column c-1 is chromosome c).
* A phenotype with two values gets a linear-probability background and a warning (no
  logistic level 1).

Step 4 for these backgrounds is the blocked engine `cg.background.gwas_blocked` (numpy or
cupy, `.bed` or cugen input, several backgrounds in one pass). It reproduces the linear
`cg.gwas` statistic (max |ΔZ| ≈ 2e-6 on public Darwin's Ark data) and writes the same columns
(`CHR POS ID REF ALT BETA SE Z P MAF gidx`) and output layout; `cg.gwas` will dispatch to it
as `engine="blocked"` in a later release. `cg.summarize(ss)` gives λ_GC, the GWS count and
1-Mb distance clumps from Z.

## Running on CPU

```python
import cugen as cg

cg.cuda.is_available()                        # cupy imports, a device exists and a 1-element op runs
geno = cg.open_genotypes("cohort")            # PLINK prefix (counts .bim A1) or a cugen directory
loco = cg.fit_cugenie(geno, y, covariates, snps="prune.prune.in", device="cpu")
ss   = cg.background.gwas_blocked(geno, loco)                 # all variants
info, loci = cg.summarize(ss)                                  # lambda_GC, GWS, loci
loco.save("out/loco_predictions.npz"); ss.to_feather("out/height.sumstats.feather")
```

* **Devices** follow PyTorch naming: `cg.device("auto" | "cpu" | "cuda" | "cuda:N")`, an int
  is a CUDA index. Every new function takes `device="auto"` (the `CUGEN_DEVICE` environment
  variable overrides "auto"). `pip install cugen` without the `gpu` extra is enough for the CPU
  path; cupy is never imported unless a GPU is used.
* **Samples and phenotypes.** `y` rows follow `geno.sample_ids`; phenotypes must be complete
  cases. For REGENIE-identical folds keep samples in `.fam` order.
* **Alleles.** `BedGenotypes(counted="A1")` counts `.bim` column 5 by default (PLINK/REGENIE;
  note plink2 writes ALT there), `counted="A2"` column 6. `CugenGenotypes` counts the cugen's
  `ALT`; `cg.bed2cugen` stores column 6 by default and takes the same `counted=` option. In the
  output `ALT` is always the counted allele, and `ss.attrs["counted_allele"]` and the log name it.
* **Resources.** Every `fit_*` and `gwas_blocked` accepts `estimate_only=True` and returns,
  without computing anything, `work_bytes` (what the call allocates), `peak_bytes` (whole
  process), FLOPs per stage and the dominant stage. Before allocating, each call checks
  `work_bytes` against free memory (the job's cgroup limit minus non-reclaimable usage, or free
  VRAM) and raises `cg.CugenResourceError` listing options that would fit (`low_memory=True`, a
  pruned `snps=`/`grm_snps=` panel, `gram_dtype="float32"`, `pairwise_count=False`, smaller
  blocks). `low_memory="auto"` (default) switches by itself; `memory_budget="6G"` or
  `CUGEN_MEMORY_LIMIT=6G` caps it. (`gpu.memory_limit_gb` in the workflow config is the CuPy pool
  limit, a different thing.)
* **Measured costs** (dog height, defaults, 4 CPU cores and an 8 GB memory limit; peak RSS from
  `/usr/bin/time`): Darwin's Ark 2022, n = 1,949, 8.86M variants; Darwin's Ark 2025, n = 3,074,
  10.1M variants.

  | stage | 2022, CPU | 2025, CPU | 2022, one GPU |
  |---|---|---|---|
  | `fit_cugenie` (500k-SNP step-1 panel, 3 phenotypes) | 8.4 min, 1.7 GiB | 8.7 min, 2.0 GiB | 41 s |
  | `fit_cugenie_k` (pruned panel, 770k / 841k SNPs) | 2.8 min, 2.7 GiB | 7.0 min, 4.8 GiB (`low_memory=True`: 10.3 min, 2.0 GiB) | 4 s |
  | `gwas_blocked`, all variants, cugen input | 5.4 min, 2.2 GiB (`.bed` input: 8.8 min, 1.9 GiB) | 5.6 min, 2.6 GiB | 2 s |
  | UltraLOCO, pruned GRM, all variants tested | 16.7 min, 4.4 GiB | 40 min, 3.9 GiB (`low_memory` chosen automatically) | - |
  | UltraLOCO, all-variant GRM | 39 min (GRM pass 25 min) | not run (`estimate_only`: ~590 TFLOP) | 26 s |

  CPU time goes to decoding the 2-bit genotypes (step 4), to the level-0 eigendecompositions
  (CuGenie), and to the GRM pass or the whitened test (UltraLOCO; use a pruned `grm_snps` panel
  on CPU).
* The UltraLasso steps 1–3 and the pool step 4 (`cg.gwas`) need CUDA.

## Demo data (`cugen.datasets`)

`cugen.datasets` ships code, not data: a generator for a fully synthetic GWAS cohort, evaluators that
score results against its known truth, and a checksum-verified `fetch()` for published datasets.
Everything is CPU-only (numpy and the standard library).

### Synthetic "exaggerated height"

```bash
python -m cugen.datasets.simulate_height --out demo_height --preset n3k  --n-workers 8   # n = 3,000, ~1 GB
python -m cugen.datasets.simulate_height --out demo_height --preset n20k --n-workers 8   # n = 20,000, ~12 GB, ~7 min
cd demo_height && python -c "import cugen as cg; cg.ultra_workflow('config_ultralasso.json')"
python -m cugen.datasets.evaluate results/ultralasso/height/sumstats.tsv.gz .
```

```python
from cugen.datasets import simulate_height, workflow_config, evaluate_gwas, background_r2
meta = simulate_height("demo_height", preset="n3k", n_workers=8)    # keyword arguments override the preset
cfg = workflow_config("demo_height", "results/std")                   # package defaults
```

LD comes from a haplotype-mosaic model (founder haplotypes per LD block, Markov switching inside a
block); `n_pops`/`fst` add Balding-Nichols structure and `strat_shift` a population mean shift;
`n_chrom` above 22 uses equal-length chromosomes (`chrom_lengths="equal"`). Outputs: per-chromosome
`.cugen`, PLINK `.bed/.bim/.fam` (A1 = ALT, the counted allele), `pheno.tsv`, `covar.tsv`,
`truth.tsv`, `annotation.feather`, `ld_blocks.tsv`, a held-out `test/` cohort, a ready
`config_ultralasso.json` and `meta.json`. The run is deterministic for a seed whatever `n_workers`;
because numpy does not promise identical random streams across versions, `meta.json` records the
numpy version and a sha256 per file, and `cugen.datasets.verify("demo_height", "demo_height_n20k")`
reports any file that differs.

On `n20k` with the package defaults, UltraLasso found 259 loci against REGENIE v2.2.4's 155, with
genomic control over causal-free LD blocks of 1.003 vs 1.002 and no false loci; 150 of REGENIE's
155 loci lie within 250 kb of an UltraLasso locus. Read that with three qualifications: the two
association statistics correlate at r(z) = 0.77 rather than ~1, because the LOCO backgrounds differ
in strength (in-sample R² 0.51 vs 0.16); at this sample size the default LOCO offset attenuates
causal effects by 7-8 %; and this is one synthetic architecture, not a benchmark across traits.
`evaluate_finemapping` scores UltraSuSiE credible sets against the truth. For small cohorts the
config uses one step-4 worker (`gpu.n_workers_gwas = 1`); scripts that use more need an
`if __name__ == "__main__":` guard.

### Published datasets

```python
from cugen import datasets
datasets.list_datasets()                       # demo_height_n20k, demo_height_n3k, darwins_ark_2022, darwins_ark_2025
datasets.info("darwins_ark_2022")              # files, sizes, sha256, licence, citation
path = datasets.fetch("darwins_ark_2022", parts=["genotypes"])
datasets.verify(path, "darwins_ark_2022")
```

Downloads resume (HTTP `Range`), are checked against the manifest's size and sha256, and land in
`dest=`, else `$CUGEN_DATA`, else `~/.cache/cugen/datasets`; point it at a large filesystem.
Archives are extracted member by member with path checks, and nothing in them is executed.

| name | contents | size | access |
|---|---|---|---|
| `demo_height_n20k` / `demo_height_n3k` | the synthetic cohorts above | ~12 GB / ~1 GB | not published yet: `fetch` raises `DatasetNotHostedError` with the regenerate command |
| `darwins_ark_2022` | Darwin's Ark genotypes and phenotypes (Morrill et al., *Science* 2022), Zenodo mirror 6522639 of Dryad doi:10.5061/dryad.g4f4qrfr0, CC0 | 7.0 GB | open |
| `darwins_ark_2025` | Darwin's Dogs data (Lord et al., *PNAS* 2025), Dryad doi:10.5061/dryad.83bk3jb4r, CC0 | 19.5 GB | manual: Dryad needs a login; download by hand, then `verify()` |

`fetch` stops at verified raw archives: deriving a phenotype and converting the genotypes are up to
the user. Note the allele conventions when you convert: `cugen.BedGenotypes` counts the bim A1
allele (as PLINK and REGENIE do), while `cugen.bed2cugen` counts A2 by default (`counted="A1"` to match).

## The `.cugen` file format

A single fixed-layout binary supporting both streaming and random
access on GPU:

```
┌─────────────────────────────────────────────────────────┐
│ Header (256 B)  magic + n_samples + n_variants          │
│                  + gidx_offset                          │
├─────────────────────────────────────────────────────────┤
│ Stats           4 × f32 per variant                     │
│                  (μₓ, Sₓₓ, MAF, _reserved)              │
├─────────────────────────────────────────────────────────┤
│ Gidx            int64 per variant (global index)        │
├─────────────────────────────────────────────────────────┤
│ Genotypes       2-bit packed per variant, big-endian    │
│                  ⌈N/4⌉ bytes/variant, fixed size        │
└─────────────────────────────────────────────────────────┘
```

I/O modes:

| Mode             | Used by                              | How                                              |
|------------------|--------------------------------------|--------------------------------------------------|
| Streaming        | ultraLasso, QC, subset, score        | Pinned host ring + async `cudaMemcpyAsync`       |
| Random access    | UltraSuSiE, LD, per-locus statistics | Direct `pread` on fixed-size records (O(1) seek) |

Streaming I/O overlaps disk reads, host-to-device transfers, and GPU
compute so the device never idles waiting for I/O. Random access lets
UltraSuSiE compute exact in-sample sufficient statistics XᵀX, Xᵀr per
locus without scanning unrelated variants.

## v0.1.3 API surface

### `cg.ultralasso.*` — the ultraLasso pipeline (steps 1–5)

| Call                       | Stage                                                     |
|----------------------------|-----------------------------------------------------------|
| `cg.ultralasso.screen`     | Step 1+2 (per-chr univariate screen + local LASSO)        |
| `cg.ultralasso.fit`        | Step 3 (joint LASSO + LOCO residuals)                     |
| `cg.ultralasso.gwas`       | Step 4 (association testing)                              |
| `cg.ultralasso.loci`       | Step 5 — locus calling                                    |
| `cg.ultralasso.susie`      | Step 5 — UltraSuSiE Tier 1 fine-mapping                   |
| `cg.ultralasso.map_`       | Step 5 — UltraMAP Tier 2 fine-mapping                     |
| `cg.ultralasso.run`        | End-to-end one-call orchestrator                          |

### `cg.background.*` — LOCO backgrounds on CPU or GPU

| Call | Stage |
|---|---|
| `cg.open_genotypes` / `cg.BedGenotypes` / `cg.CugenGenotypes` | genotype block readers (.bed, cugen) |
| `cg.fit_cugenie`, `cg.fit_cugenie_k`, `cg.fit_ultraloco` | backgrounds (`LocoPredictions`) |
| `cg.background.fit_joint_lasso` | UltraLasso (same as `cg.fit_joint_lasso`) |
| `cg.background.gwas_blocked`, `cg.ultraloco_gwas` | step 4 for any background; UltraLOCO's GLS test |
| `cg.summarize`, `cg.lambda_gc` | λ_GC, GWS count, distance clumps |
| `cg.device`, `cg.cuda` | device selection and CUDA queries |

### Workflow + data layer

`cg.ultra_workflow` (JSON-driven orchestrator) · `cg.prepare_cohort` ·
`cg.read_cugen` · `cg.CugenReader`

### General toolkit

`cg.frequency` · `cg.variant_qc` · `cg.sample_qc` · `cg.score` ·
`cg.subset_cugen_file` / `cg.subset_cugen_dir` · `cg.plot.qq` ·
`cg.plot.manhattan`

### PLINK / Hail aliases

| PLINK / plink2  | Hail                          | CuGen              |
|-----------------|-------------------------------|--------------------|
| `--keep`        | `mt.filter_cols`              | `keep`             |
| `--extract`     | `mt.filter_rows`              | `extract`          |
| `--remove`      |                               | `remove`           |
| `--freq`        |                               | `freq`             |
| `--glm`         | `linear_regression_rows`      | `glm`              |
| `--score`       |                               | `score`            |
| `--clump`       |                               | `clump` *(v0.2)*   |
| `--king`        | `realized_relationship_matrix`| `make_king` *(v0.2)* |
|                 | `hwe_normalized_pca`          | `pca` *(v0.2)*     |

## Workflow JSON

A single config drives the full pipeline. Set `"run": false` on any
stage to skip it. The full key reference (with defaults) is in
[`cugen/config.py`](cugen/config.py).

```python
import cugen as cg

# Emit a fully-populated template you can customise:
cg.write_example_config("example_config.json")
```

CLI entry point:

```bash
cugen workflow config.json                # run full pipeline
cugen workflow --example > config.json    # emit template
cugen info data/chr22.cugen               # header + stats
cugen freq data/chr22.cugen               # allele frequencies
```

## Extending CuGen — bring your own GPU code

CuGen is fully [CuPy](https://cupy.dev)-native. Every reader, statistic
and pipeline stage hands you live `cupy.ndarray` objects you can plug
straight into your own GPU code — write a custom estimator, a new QC
test, an alternative strength function, a different fine-mapper — and
mix it in alongside the built-in functions without leaving the device:

```python
import cupy as cp
import cugen as cg

# 1) Drop-in custom strength function for step 1+2
from cugen.screen import register_strength_fn

@register_strength_fn("my_strength")
def my_strength(stats):
    # stats has slope / intercept / maf / sxx / syy / n / mu_x as
    # numpy arrays — return a 1-D strength vector of your choice
    return (stats["slope"] ** 2) * cp.log1p(stats["sxx"]).get()

cg.ultralasso.screen(..., strength_fn="my_strength")

# 2) Read a chunk of .cugen straight to GPU and run your own kernel
reader = cg.read_cugen("data/chr22.cugen")          # auto-pinned if USE_PINNED_READER=1
X_gpu = reader.read_to_gpu(0, 5000)                 # (n_samples, 5000) on device
my_score = cp.einsum("nv,n->v", X_gpu, y_gpu) / X_gpu.shape[0]

# 3) Compose any of cg.* with your own functions in a single script
sumstats = cg.ultralasso.gwas(...)
my_followup = my_gpu_function(cp.asarray(sumstats["beta"].values))
```

There is no plugin manifest and no callback API — the package is just
Python + CuPy. Read a `.cugen` file, get GPU arrays, do whatever you
want with them, and feed results back into the downstream stages.
Built-ins are starting points, not constraints.

## License

MIT © 2026 Tuomo Kiiskinen. See [LICENSE](LICENSE).

## Citing

Please cite the CuGen preprint (forthcoming) when using this software:

> Kiiskinen, T., Richland, J., Wang, W., Lu, S., Narasimhan, B., Hastie,
> T., Tibshirani, R., Rivas, M.A. *CuGen: a GPU-accelerated framework
> for large-scale genomics.* Preprint forthcoming.

And the underlying methods:

- Chatterjee, A., et al. *Univariate-guided sparse regression
  (uniLasso).* 2025.
- Wang, G., et al. *A simple new approach to variable selection in
  regression, with application to genetic fine-mapping.* JRSS-B, 2020.

## Status

v0.1.3 — Beta. The framework is in active development. APIs may still
shift between minor versions. Bug reports + feature requests welcome
on the [issue tracker](https://github.com/yuj1r0/cugen/issues).
