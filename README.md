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

Requires Python ≥ 3.10, CUDA 12.x, and a recent NVIDIA GPU
(80 GB device memory recommended at biobank scale).

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
