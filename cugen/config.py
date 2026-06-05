"""cugen.config — JSON schema, validation, and defaults for ultra_workflow."""

import json
from pathlib import Path
from typing import Any, Dict, Optional, Union

DEFAULTS: Dict[str, Any] = {
    "_version": "1.0",
    "general": {
        "name": "run",
        "output_dir": "results/run",
        "device": 0,
        "seed": 42,
        "log_level": "INFO",
    },
    "gpu": {
        "memory_limit_gb": 70,
        "use_pinned_reader": True,
        "n_workers_gwas": 8,
        "n_workers_finemapping": 8,
        "cuda_mps": True,
    },
    "data": {
        # Default cugen directory used by every stage unless that stage
        # supplies its own ``cugen_dir`` override (see step4_gwas /
        # step5b_finemapping). Per-chromosome files are looked up as
        # ``cugen_pattern.format(chrom=...)`` (default ``chr{chrom}.cugen``).
        "cugen_dir": None,
        "cugen_pattern": "chr{chrom}.cugen",
        "chromosomes": list(range(1, 23)),
        # Path to gidx annotation feather (gidx / CHR / POS / ID / REF /
        # ALT). Required by step 4 (to fill in genome coords) and step 5b
        # (locus annotation). ``annotation`` is canonical;
        # ``annotation_file`` kept as legacy alias.
        "annotation": None,
        "annotation_file": None,
        # Path to master phenotype TSV (FID + covariates + traits). Step 3
        # uses this to assemble the covariate matrix.
        "phe_file": None,
    },
    "phenotype": {
        "mode": "single",
        "file": None,
        "name": None,
        "type": "continuous",
        "phenotype_list": None,
    },
    # Step 0: residual-cohort build. When the workflow needs a cohort NPZ
    # that doesn't exist yet, it runs the single quick OLS residualisation
    # of cugen.prep_cohort.prepare_cohort here, in-pipeline. ``build``:
    #   "auto" (default) build only if ``npz`` (or phenotype.file) is missing,
    #   True  always rebuild, False never build (use the existing NPZ).
    # ``ref_npz`` is the base cohort defining the canonical .cugen sample
    # order; ``psam``/``n_samples`` come from the genotype set; ``phenotype``
    # is the master.phe column. master.phe is taken from data.phe_file and
    # covariates from covariates.columns.
    "cohort": {
        "build": "auto",
        "phenotype": None,
        "ref_npz": None,
        "psam": None,
        "n_samples": None,
        "binary": None,      # None = auto-detect from value set
        "npz": None,         # output/input path (overrides phenotype.file)
    },
    "covariates": {
        "columns": [
            "age", "sex",
            "PC1", "PC2", "PC3", "PC4",
            "PC5", "PC6", "PC7", "PC8",
            "PC9", "PC10",
        ],
    },
    "step1_screening": {
        "run": True,
        # Variants per local-LASSO window (the screening block).
        # Legacy alias ``block_size`` is also accepted in user configs.
        "window_size": 8192,
        # Per-window keep fraction (%) handed to the step-2 local LASSO.
        # Legacy alias ``block_pct`` is also accepted in user configs.
        "top_pct": 20.0,
        # MAF floor on the per-variant screen. PRODUCTION value is 1e-3
        # (run_ultralasso_pipeline_pipelined.sh passes no --maf-min, so
        # step1_to_step2 uses its 0.001 default). A higher floor drops the
        # MAF∈[1e-3,1e-2) candidates that fill out the active set: at 1e-2
        # the fwb INI50 active set fell 22,301 -> 18,201 (-18%), weakening
        # LOCO confounding control. Keep 1e-3 to replicate the manuscript.
        "maf_min": 1e-3,
        # TopK ranking function. Built-ins: 'r2_se' (default, production),
        # 'r2', 'beta_abs', 'z_abs', 'chi2'. Or register your own via
        # cugen.screen.register_strength_fn().
        "strength_fn": "r2_se",
        # Per-chromosome-group chunking for the streaming screen pass.
        # Each ``split_chrs`` entry is an inclusive 1-based range "lo:hi";
        # ``split_chrs_by[i]`` is the number of equal variant chunks for that
        # group (chunked at the cugen-header variant midpoints, exactly like
        # the production half-split). Any chromosome not covered by a range
        # uses ``default_chunks``. PRODUCTION DEFAULT: chr 1-11 -> 2 chunks
        # (the half-split that keeps a 408K-sample chr1 within 80 GB),
        # chr 12-22 -> whole. Users with larger data / smaller GPUs can raise
        # the chunk counts, e.g. ["1:11","12:22"] / [4,2].
        "split_chrs": ["1:11", "12:22"],
        "split_chrs_by": [2, 1],
        "default_chunks": 1,
    },
    "step2_local_lasso": {
        "run": True,
        "alpha": 6e-4,
        "max_iter": 1000,
        "elastic_net_ratio": 0.99,
    },
    "step3_global_lasso": {
        "run": True,
        "alpha": 6e-3,
        "ridge": 1e-2,
        "max_iter": 2000,
        "elastic_net_ratio": 0.99,
        "compute_loco": True,
        "loco_mode": "ols",
        "n_loco": 22,
        "precomputed_loco": None,
        "precomputed_active_set": None,
        # Cap the candidates fed into the joint LASSO to the top-N by
        # ``max_snps_rank_by`` (default the screen 'strength' = r²/SE²; 'r2'
        # also accepted). Decouples step-2 alpha from the step-3 GPU budget:
        # a loose alpha can over-select, and this chainsaw trims to a size
        # that fits (≈38K is the proven step-3 ceiling at 408K samples on an
        # 80 GB GPU — beyond it the LOCO solve OOMs). None = no cap.
        "max_snps": None,
        "max_snps_rank_by": "strength",
    },
    "step4_gwas": {
        "run": True,
        "test_imputed": True,
        "test_array": True,
        "test": "wald",
        "family": "linear",        # 'binary' -> logistic score test + selective SPA
        "spa_threshold": 5.0,
        "maf_min_test": 1e-3,
        # Binary-trait (family='binary') SPA controls. min_mac is the standard
        # biobank post-hoc minor-allele-count floor (LOW_MAC below it); SPA is
        # applied where the normal tail is suspect (p_norm<spa_pthresh & |Z|>sd).
        "min_mac": 20.0,
        "spa_pthresh": 0.05,
        "spa_sd_thresh": 2.0,
        "spa_chunk": 64,
        # Optional per-stage overrides. ``cugen_dir``/``annotation`` let
        # step 4 GWAS run against a DIFFERENT variant set than steps 1-3
        # (e.g. steps 1-3 on array, step 4 on imputed). None -> fall back to
        # data.cugen_dir / data.annotation.
        "cugen_dir": None,
        "annotation": None,
    },
    # Step 5a: locus definition (split out from old step5_finemapping).
    "step5_finemapping": {
        "run": True,
        "method": "ultrasusie",  # legacy; superseded by step5b_finemapping.tier
        "n_signals": 10,
        "p_threshold": 5e-8,
        "window_kb": 500,
        "merge_overlapping": True,
        "cache_xtx": True,
    },
    # Step 5b: fine-mapping (UltraSuSiE Tier 1 / UltraMAP Tier 2).
    # Driven from a single JSON config; runs as the final stage of the
    # UltraLasso pipeline after step5_finemapping defines loci.
    "step5b_finemapping": {
        "run": True,
        "tier": 1,                # 1 = UltraSuSiE; 2 = UltraMAP
        "n_signals": 10,          # SuSiE L
        "n_workers": 8,           # 1 = sequential; 8 = production H100 SXM5
        "coverage": 0.95,
        "max_variants": 15000,
        # Tier 2 (UltraMAP) extras — ignored when tier=1:
        "adaptive_weight_method": "z",   # 'z' or 'strength'
        "adaptive_weight_gamma": 1.0,
        "lasso_cs_level": 50,            # 50..90
        "lambda_min_ratio": 0.01,
        "gwas_sumstats_dir": None,       # for adaptive LASSO; defaults to step4 dir
        # Optional per-stage overrides (same idea as step4_gwas). Fine-mapping
        # runs against whichever variant set step 4 GWAS'd. None -> fall back
        # to data.cugen_dir / data.annotation.
        "cugen_dir": None,
        "annotation": None,
        # When set, run a SECOND timed fine-mapping pass on the top-N most
        # significant loci only (by lead_p), to report a normalised
        # "seconds per 1000 loci" metric alongside the full genome-wide run.
        # None = skip the capped pass. Outputs land in finemapping_top/.
        "benchmark_max_loci": 1000,
    },
    "qc": {
        "run_variant_qc": False,
        "run_sample_qc": False,
        "maf_filter": None,
        "geno_filter": None,
        "hwe_filter": None,
        "mind_filter": None,
    },
    "output": {
        "save_sumstats": True,
        "save_loco_predictions": False,
        "save_active_set": True,
        "save_finemapping": True,
        "sumstats_format": "tsv.gz",
        "plots": {
            "manhattan": True,
            "qq": True,
            "finemapping_loci": True,
        },
    },
}


REQUIRED_FIELDS = [
    ("general", "name"),
    ("general", "output_dir"),
    ("data", "cugen_dir"),
    ("phenotype", "mode"),
]


def _deep_merge(defaults: dict, override: dict) -> dict:
    """Recursive dict merge: override values win, but missing keys fall to defaults."""
    out = dict(defaults)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path_or_dict: Union[str, Path, dict]) -> dict:
    """Load a JSON config (path or dict) and merge over DEFAULTS."""
    if isinstance(path_or_dict, (str, Path)):
        with open(path_or_dict, "r") as f:
            user = json.load(f)
    elif isinstance(path_or_dict, dict):
        user = path_or_dict
    else:
        raise TypeError(f"Expected str/Path/dict, got {type(path_or_dict).__name__}")
    return _deep_merge(DEFAULTS, user)


def validate_config(cfg: dict) -> dict:
    """Validate a merged config dict. Raises ValueError on missing required fields.

    Returns the (possibly normalised) config dict.
    """
    errors = []
    for section, key in REQUIRED_FIELDS:
        if section not in cfg or cfg[section].get(key) in (None, ""):
            errors.append(f"required field missing: {section}.{key}")

    pheno = cfg.get("phenotype", {})
    if pheno.get("mode") == "single":
        if not pheno.get("file"):
            errors.append("phenotype.mode=='single' requires phenotype.file")
    elif pheno.get("mode") == "batch":
        if not pheno.get("phenotype_list"):
            errors.append("phenotype.mode=='batch' requires phenotype.phenotype_list")
    elif pheno.get("mode") is not None:
        errors.append(f"unsupported phenotype.mode={pheno.get('mode')!r}")

    if cfg["step5_finemapping"]["method"] not in ("ultrasusie", "ultramap"):
        errors.append("step5_finemapping.method must be 'ultrasusie' or 'ultramap'")

    # step1_screening chromosome-split config
    s1 = cfg.get("step1_screening", {})
    split_chrs = s1.get("split_chrs", [])
    split_by = s1.get("split_chrs_by", [])
    if len(split_chrs) != len(split_by):
        errors.append(
            f"step1_screening.split_chrs (len {len(split_chrs)}) and "
            f"split_chrs_by (len {len(split_by)}) must have equal length"
        )
    for rng in split_chrs:
        try:
            lo, hi = str(rng).split(":")
            lo, hi = int(lo), int(hi)
            if not (1 <= lo <= hi <= 22):
                errors.append(f"step1_screening.split_chrs range {rng!r} "
                              "must satisfy 1<=lo<=hi<=22")
        except (ValueError, AttributeError):
            errors.append(f"step1_screening.split_chrs entry {rng!r} must be "
                          "'lo:hi' (e.g. '1:11')")
    for k in split_by:
        if not (isinstance(k, int) and k >= 1):
            errors.append(f"step1_screening.split_chrs_by entry {k!r} must be "
                          "an int >= 1")

    # step5b benchmark_max_loci
    bm = cfg.get("step5b_finemapping", {}).get("benchmark_max_loci")
    if bm is not None and not (isinstance(bm, int) and bm > 0):
        errors.append("step5b_finemapping.benchmark_max_loci must be a "
                      "positive int or null")

    if errors:
        raise ValueError("Config validation failed:\n  - " + "\n  - ".join(errors))
    return cfg


def load_and_validate_config(path_or_dict: Union[str, Path, dict]) -> dict:
    return validate_config(load_config(path_or_dict))


def write_example_config(path: Union[str, Path]) -> None:
    """Write a fully-populated example config (all keys, defaults) to ``path``."""
    with open(path, "w") as f:
        json.dump(DEFAULTS, f, indent=4, default=str)
