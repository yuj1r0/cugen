"""cugen.workflow — JSON-driven orchestrator for the full UltraLasso pipeline.

The single entry point ``ultra_workflow(config)`` accepts either:
  - a path to a JSON config file
  - a dict (already-parsed config)

It loads + validates the config, sets up the GPU + output directory, and
dispatches to:
    prepare_cohort  (if phenotype.file points at master.phe — currently
                     consumers should pre-build the residual cohort NPZ)
    screen_chromosome  (Step 1+2)
    fit_joint_lasso    (Step 3)
    gwas               (Step 4)
    define_loci        (Step 5a)
    ultrasusie/ultramap (Step 5b)

Single-phenotype + batch modes are supported. Partial runs are supported by
setting ``"run": false`` on any stage. v0.1 ships the orchestration skeleton;
underlying step modules raise NotImplementedError until v0.1.1 (see module
docstrings).
"""

import json
import logging
import time
from pathlib import Path
from typing import Optional, Union

import pandas as pd

from .config import load_and_validate_config


def _setup_logging(level: str = "INFO") -> logging.Logger:
    log = logging.getLogger("cugen")
    if not log.handlers:
        log.setLevel(getattr(logging, level.upper(), logging.INFO))
        h = logging.StreamHandler()
        h.setFormatter(logging.Formatter(
            "[%(asctime)s][cugen][%(levelname)s] %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        log.addHandler(h)
    return log


def _setup_gpu(gpu_cfg: dict, device: int) -> None:
    """Best-effort GPU device + memory-pool setup. Silent no-op when GPU absent.

    Note: ``memory_limit_gb`` is the *parent* process budget. Step 4 GWAS
    and step 5b fine-mapping fan out into Pool(spawn) workers, each of
    which sets its OWN pool limit inside the worker — we don't apply the
    limit here when multiple GPU workers are configured, so we don't have
    to halve / quarter it ourselves. If a single-process pipeline is
    requested (no fan-out), the parent uses the full limit.
    """
    try:
        import cupy as cp
        cp.cuda.Device(device).use()
        mem_limit_gb = gpu_cfg.get("memory_limit_gb")
        fan_out = max(int(gpu_cfg.get("n_workers_gwas", 1) or 1),
                      int(gpu_cfg.get("n_workers_finemapping", 1) or 1))
        if mem_limit_gb and fan_out <= 1:
            pool = cp.get_default_memory_pool()
            pool.set_limit(size=int(mem_limit_gb * 1024 ** 3))
    except Exception:
        # CPU-only environments (login node, no CUDA driver) still allow
        # validation + I/O paths and dry_run.
        pass


def _resolve_phenotypes(pheno_cfg: dict, output_dir: Path) -> list:
    """Return list of (name, file, type) tuples to iterate."""
    mode = pheno_cfg.get("mode", "single")
    if mode == "single":
        return [(
            pheno_cfg.get("name") or Path(pheno_cfg["file"]).stem,
            pheno_cfg["file"],
            pheno_cfg.get("type", "continuous"),
        )]
    elif mode == "batch":
        tsv = pd.read_csv(pheno_cfg["phenotype_list"], sep="\t")
        cols = {c.lower() for c in tsv.columns}
        need = {"name", "file", "type"}
        if not need.issubset(cols):
            raise ValueError(
                f"phenotype_list TSV must have columns {need}, got {set(tsv.columns)}"
            )
        return list(tsv[["name", "file", "type"]].itertuples(index=False, name=None))
    else:
        raise ValueError(f"unsupported phenotype.mode={mode}")


def _parse_chr_range(rng: str) -> range:
    """'1:11' -> range(1, 12) (inclusive hi)."""
    lo, hi = str(rng).split(":")
    return range(int(lo), int(hi) + 1)


def _build_chunk_count_map(s1: dict) -> dict:
    """Map chromosome int -> chunk count from step1_screening split config.

    Chromosomes not covered by any ``split_chrs`` range get ``default_chunks``
    (default 1 = whole-chromosome). Later ranges win on overlap.
    """
    default = int(s1.get("default_chunks", 1))
    out = {c: default for c in range(1, 23)}
    ranges = s1.get("split_chrs", []) or []
    counts = s1.get("split_chrs_by", []) or []
    for rng, k in zip(ranges, counts):
        for c in _parse_chr_range(rng):
            out[c] = int(k)
    return out


def _chunk_bounds(n_variants: int, k: int) -> list:
    """k contiguous [start, end) chunks over [0, n_variants), integer split.

    ``k==1`` -> [(0, n_variants)]. ``_chunk_bounds(n, 2)`` -> [(0, n//2),
    (n//2, n)], byte-identical to the production ``MIDPOINT = n // 2`` split.
    Empty chunks (when k > n_variants) are dropped.
    """
    if k <= 1:
        return [(0, n_variants)]
    edges = [(n_variants * i) // k for i in range(k + 1)]
    return [(edges[i], edges[i + 1]) for i in range(k)
            if edges[i + 1] > edges[i]]


def ultra_workflow(config: Union[str, Path, dict],
                   dry_run: bool = False) -> dict:
    """Run the full UltraLasso pipeline from a JSON config.

    Parameters
    ----------
    config : path-to-json or dict
    dry_run : if True, only validate + plan, do not execute any stage.

    Returns
    -------
    dict
        Per-phenotype summary with paths to outputs + timings.
    """
    t_start = time.time()
    cfg = load_and_validate_config(config)
    log = _setup_logging(cfg["general"]["log_level"])
    log.info("ultra_workflow start (v0.1)")

    output_dir = Path(cfg["general"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_dir / "_config.resolved.json", "w") as f:
        json.dump(cfg, f, indent=2, default=str)
    log.info(f"resolved config → {output_dir / '_config.resolved.json'}")

    if not dry_run:
        _setup_gpu(cfg["gpu"], cfg["general"]["device"])

    phenotypes = _resolve_phenotypes(cfg["phenotype"], output_dir)
    log.info(f"resolved {len(phenotypes)} phenotype(s)")

    if dry_run:
        log.info("dry_run=True — skipping execution")
        return {
            "config": cfg,
            "phenotypes": phenotypes,
            "elapsed_s": time.time() - t_start,
            "dry_run": True,
        }

    # Lazy imports — keeps `ultra_workflow(dry_run=True)` callable on a login
    # node without CuPy installed.
    from .assoc import gwas as _gwas
    from .finemapping import ultramap, ultrasusie
    from .lasso import fit_joint_lasso as _fit_joint_lasso
    from .loci import define_loci as _define_loci
    from .screen import screen_chromosome as _screen_chromosome

    results = {}
    for pheno_name, pheno_file, pheno_type in phenotypes:
        log.info(f"--- phenotype: {pheno_name} ({pheno_type}) -----------------")
        pheno_dir = output_dir / pheno_name
        pheno_dir.mkdir(parents=True, exist_ok=True)
        per_stage_t = {}

        # --- Step 0: residual-cohort build (single quick OLS) ------------
        # The cohort NPZ defaults to phenotype.file; cohort.npz overrides it.
        # When the NPZ is missing (build="auto") or build=True, build it
        # in-pipeline via prepare_cohort so the JSON is self-contained
        # (pheno code in -> sumstats out).
        cohort_cfg = cfg.get("cohort", {}) or {}
        cohort_npz = cohort_cfg.get("npz") or pheno_file
        _build = cohort_cfg.get("build", "auto")
        _need_build = (_build is True) or (
            _build == "auto" and cohort_npz is not None
            and not Path(cohort_npz).exists())
        if _need_build:
            t = time.time()
            from .prep_cohort import prepare_cohort as _prepare_cohort
            _prepare_cohort(
                phenotype=cohort_cfg["phenotype"],
                ref_npz=cohort_cfg["ref_npz"],
                master_phe=cfg["data"].get("phe_file")
                           or cfg["data"].get("master_phe"),
                psam=cohort_cfg["psam"],
                n_samples=cohort_cfg["n_samples"],
                covariates=cfg["covariates"]["columns"],
                binary=cohort_cfg.get("binary"),
                output=cohort_npz,
            )
            per_stage_t["step0_cohort"] = time.time() - t
            log.info(f"step0 cohort build done in "
                     f"{per_stage_t['step0_cohort']:.1f}s -> {cohort_npz}")
        elif cohort_cfg.get("phenotype"):
            log.info(f"step0 cohort: using existing NPZ {cohort_npz}")

        # --- Step 1+2: screening + local LASSO ---------------------------
        if cfg["step1_screening"]["run"] and cfg["step2_local_lasso"]["run"]:
            from .io import read_cugen_header
            t = time.time()
            s1 = cfg["step1_screening"]
            window_size = s1.get("window_size", s1.get("block_size", 8192))
            top_pct = s1.get("top_pct", s1.get("block_pct", 20.0))
            chunk_map = _build_chunk_count_map(s1)
            candidates = []
            for chrom in cfg["data"]["chromosomes"]:
                chrom = int(chrom)
                cugen_path = Path(cfg["data"]["cugen_dir"]) / \
                    cfg["data"]["cugen_pattern"].format(chrom=chrom)
                # Production splits chr 1-11 into halves at the cugen-header
                # variant midpoint so a 408K-sample chr1 fits in 80 GB during
                # screening (see run_ultralasso_pipeline_pipelined.sh:149-164).
                k = chunk_map.get(chrom, 1)
                if k > 1:
                    n_variants = int(
                        read_cugen_header(str(cugen_path))["n_variants"])
                    bounds = _chunk_bounds(n_variants, k)
                else:
                    bounds = [(0, None)]  # whole chromosome, no range filter
                for ci, (v_start, v_end) in enumerate(bounds):
                    suffix = f"_{ci + 1}" if k > 1 else ""
                    df = _screen_chromosome(
                        chrom=chrom,
                        cohort_npz=cohort_npz,
                        cugen_path=cugen_path,
                        block_size=window_size,
                        block_pct=top_pct,
                        alpha=cfg["step2_local_lasso"]["alpha"],
                        maf_min=s1.get("maf_min", 1e-2),
                        strength_fn=s1.get("strength_fn", "r2_se"),
                        output=pheno_dir / f"chr{chrom}_screen.feather",
                        output_suffix=suffix,
                        variant_start=v_start,
                        variant_end=v_end,
                    )
                    # screen_chromosome doesn't tag rows with the chromosome
                    # it came from; step 3 (fit_joint_lasso) requires it. All
                    # chunks of one chr share chr_num=chrom, so step 3's
                    # groupby('chr_num') reunites them — suffix is irrelevant
                    # to the in-memory concat (no feather re-glob).
                    if "chr_num" not in df.columns:
                        df = df.assign(chr_num=chrom)
                    candidates.append(df)
                    # Drain the parent pool between chunks. screen_chromosome
                    # drains per-block internally but leaves its final LASSO
                    # pool cached on return; without this the next chunk
                    # competes for the VRAM the split exists to free.
                    try:
                        import cupy as _cp
                        _cp.get_default_memory_pool().free_all_blocks()
                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:  # noqa: BLE001
                        pass
            candidates = pd.concat(candidates, ignore_index=True)
            # step3 budget cap (chainsaw): a loose step-2 alpha can over-select
            # genome-wide (the chr1_1→total ratio is phenotype-dependent), and
            # >~38K candidates OOMs the step-3 LOCO solve at 408K samples. Trim
            # to the top-N by strength (r²/SE²) or r² (β²·sxx, no SE weighting).
            max_snps = cfg["step3_global_lasso"].get("max_snps")
            if max_snps and len(candidates) > int(max_snps):
                n0 = len(candidates)
                rank_by = cfg["step3_global_lasso"].get(
                    "max_snps_rank_by", "strength")
                if rank_by == "r2" and {"beta", "den"} <= set(candidates.columns):
                    candidates = (candidates
                                  .assign(_r2=candidates["beta"] ** 2
                                          * candidates["den"])
                                  .nlargest(int(max_snps), "_r2")
                                  .drop(columns="_r2")
                                  .reset_index(drop=True))
                else:
                    rank_by = "strength"
                    candidates = (candidates
                                  .nlargest(int(max_snps), "strength")
                                  .reset_index(drop=True))
                log.info(f"step3 cap: {n0} -> {len(candidates)} candidates "
                         f"(top by {rank_by})")
            per_stage_t["step12"] = time.time() - t
            log.info(f"step1+2 done in {per_stage_t['step12']:.1f}s")
        else:
            log.info("step1+2 skipped (run=False)")
            candidates = None

        # --- Step 3: joint LASSO + LOCO ----------------------------------
        if cfg["step3_global_lasso"]["run"]:
            t = time.time()
            res3 = _fit_joint_lasso(
                candidates=candidates,
                cohort_npz=cohort_npz,
                cugen_dir=cfg["data"]["cugen_dir"],
                covariates=cfg["covariates"]["columns"],
                alpha=cfg["step3_global_lasso"]["alpha"],
                ridge=cfg["step3_global_lasso"]["ridge"],
                loco_mode=cfg["step3_global_lasso"]["loco_mode"],
                phe_file=cfg["data"].get("phe_file") or cfg["data"].get("master_phe"),
                output_dir=pheno_dir,
            )
            active_set = res3["active_set"]
            loco = res3["loco_predictions"]
            per_stage_t["step3"] = time.time() - t
            log.info(f"step3 done in {per_stage_t['step3']:.1f}s")
        else:
            active_set = None
            loco = cfg["step3_global_lasso"].get("precomputed_loco")
            log.info("step3 skipped (run=False), using precomputed_loco")

        # --- Step 4: GWAS ------------------------------------------------
        sumstats = None
        if cfg["step4_gwas"]["run"]:
            # Drain the parent process's CuPy pool before fanning out to
            # multi-process GWAS workers. Without this, the parent keeps
            # ~16 GB cached from step 3 even after numpy results are
            # returned, which the workers then compete with for physical
            # GPU memory and OOM at scale (caught in workflow smoke).
            try:
                import cupy as _cp
                _cp.get_default_memory_pool().free_all_blocks()
                _cp.get_default_pinned_memory_pool().free_all_blocks()
            except Exception:  # noqa: BLE001
                pass
            t = time.time()
            # Per-stage cugen_dir/annotation override lets step 4 GWAS run
            # against a different variant set than steps 1-3 (e.g. array
            # screen -> imputed GWAS). Falls back to data.* when unset.
            gwas_dir = Path(cfg["step4_gwas"].get("cugen_dir")
                            or cfg["data"]["cugen_dir"])
            gwas_annot = (cfg["step4_gwas"].get("annotation")
                          or cfg["data"].get("annotation")
                          or cfg["data"].get("annotation_file"))
            _s4 = cfg["step4_gwas"]
            sumstats = _gwas(
                cugen_dir=gwas_dir,
                cohort_npz=cohort_npz,
                loco_predictions=loco,
                n_workers=cfg["gpu"]["n_workers_gwas"],
                family=_s4["family"],
                test=_s4["test"],
                maf_min=_s4["maf_min_test"],
                annotation=gwas_annot,
                output=pheno_dir / f"sumstats.{cfg['output']['sumstats_format']}",
                # Binary (family='binary') score+SPA controls; ignored for linear.
                min_mac=_s4.get("min_mac", 20.0),
                spa_pthresh=_s4.get("spa_pthresh", 0.05),
                spa_sd_thresh=_s4.get("spa_sd_thresh", 2.0),
                spa_chunk=_s4.get("spa_chunk", 64),
            )
            per_stage_t["step4"] = time.time() - t
            log.info(f"step4 done in {per_stage_t['step4']:.1f}s")
        else:
            log.info("step4 skipped (run=False)")

        # --- Step 5a: locus definition -----------------------------------
        loci_df = None
        if cfg["step5_finemapping"]["run"]:
            t = time.time()
            # Always re-read from disk: _gwas() returns an in-memory df
            # only when called with output=None. When the workflow passes
            # a file output, _gwas returns an empty df (line 486 in assoc.py)
            # and the actual sumstats live on disk under the file name we
            # asked for. Read that file directly.
            sumstats_path = (pheno_dir /
                f"sumstats.{cfg['output']['sumstats_format']}")
            if not sumstats_path.exists():
                # fall back to the legacy all_sumstats.tsv.gz from production
                alt = pheno_dir / "all_sumstats.tsv.gz"
                sumstats_path = alt if alt.exists() else sumstats_path
            loci_df = _define_loci(
                sumstats=str(sumstats_path),
                p_threshold=cfg["step5_finemapping"]["p_threshold"],
                window_kb=cfg["step5_finemapping"]["window_kb"],
                merge_overlapping=cfg["step5_finemapping"]["merge_overlapping"],
                output=pheno_dir / "loci.tsv",
            )
            per_stage_t["step5a"] = time.time() - t
            log.info(f"step5a done in {per_stage_t['step5a']:.1f}s, "
                     f"{len(loci_df)} loci")

        # --- Step 5b: fine-mapping ---------------------------------------
        fm_cfg = cfg.get("step5b_finemapping",
                         cfg["step5_finemapping"])  # back-compat fallback
        if fm_cfg.get("run", False) and loci_df is not None:
            t = time.time()
            # Multi-worker fine-mapping requires NPZ + feather paths (workers
            # spawn-pickle then reload). Prefer paths when available; fall back
            # to in-memory loco when n_workers <= 1.
            n_fm_workers = (cfg.get("gpu", {}).get("n_workers_finemapping")
                            or fm_cfg.get("n_workers", 1))
            loco_for_fm = loco
            if isinstance(loco, (str, Path)):
                # step3 was skipped and a precomputed_loco PATH was supplied
                # (e.g. reusing steps 1-3 across an array + imputed pass).
                # Workers can np.load it directly.
                loco_for_fm = str(loco)
            elif n_fm_workers > 1:
                loco_npz_path = pheno_dir / "loco_predictions.npz"
                if loco_npz_path.exists():
                    loco_for_fm = str(loco_npz_path)
                else:
                    log.warning(
                        f"finemap multi-worker requested ({n_fm_workers}) but "
                        f"{loco_npz_path} not on disk; falling back to "
                        f"sequential (n_workers=1)"
                    )
                    n_fm_workers = 1

            tier = int(fm_cfg.get("tier", 1))
            fm_fn = ultrasusie if tier == 1 else ultramap
            fm_out_dir = pheno_dir / "finemapping"
            # Step 5b operates on whichever cugen step 4 GWAS'd against.
            # Per-stage cugen_dir/annotation override (same idea as step4);
            # falls back to data.* when unset.
            fm_cugen_dir = (fm_cfg.get("cugen_dir")
                            or cfg["data"]["cugen_dir"])
            fm_annot = (fm_cfg.get("annotation")
                        or cfg["data"]["annotation"])
            common_kw = dict(
                loci=loci_df,
                loco_predictions=loco_for_fm,
                cohort_npz=cohort_npz,
                cugen_dir=fm_cugen_dir,
                annotation=fm_annot,
                output_dir=fm_out_dir,
                n_signals=fm_cfg.get("n_signals", 10),
                n_workers=n_fm_workers,
                coverage=fm_cfg.get("coverage", 0.95),
                max_variants=fm_cfg.get("max_variants", 15000),
            )
            if tier == 2:
                common_kw.update(
                    gwas_sumstats_dir=fm_cfg.get("gwas_sumstats_dir"),
                    adaptive_weight_method=fm_cfg.get(
                        "adaptive_weight_method", "z"),
                    adaptive_weight_gamma=fm_cfg.get(
                        "adaptive_weight_gamma", 1.0),
                    lasso_cs_level=fm_cfg.get("lasso_cs_level", 50),
                    lambda_min_ratio=fm_cfg.get("lambda_min_ratio", 0.01),
                )
            fm_fn(**common_kw)
            per_stage_t["step5b"] = time.time() - t
            log.info(f"step5b ({'UltraSuSiE' if tier == 1 else 'UltraMAP'}) "
                     f"done in {per_stage_t['step5b']:.1f}s")

            # --- Step 5b-capped: timed pass on top-N loci ----------------
            # Full step5b above is the scientific output. This second pass
            # fine-maps only the top-N most significant loci to report a
            # normalised "seconds per 1000 loci" metric. Skipped when
            # benchmark_max_loci is null.
            bm_n = fm_cfg.get("benchmark_max_loci")
            if bm_n and len(loci_df) <= int(bm_n):
                # The full pass already fine-mapped <= N loci, so a "capped"
                # pass would re-do the identical work — and a back-to-back
                # second 8-worker finemap runs markedly slower (GPU pool /
                # MPS state degrades after the first). Reuse the full timing.
                n_full = len(loci_df)
                per_stage_t["step5b_capped"] = per_stage_t["step5b"]
                per_stage_t["step5b_capped_n_loci"] = n_full
                per_stage_t["step5b_s_per_1k_loci"] = (
                    per_stage_t["step5b"] / n_full * 1000.0 if n_full else 0.0)
                log.info(
                    f"step5b capped pass: {n_full} loci <= cap {int(bm_n)}, "
                    f"reusing full-pass timing "
                    f"({per_stage_t['step5b_s_per_1k_loci']:.2f} s / 1k loci)")
            elif bm_n:
                loci_capped = _define_loci(
                    sumstats=str(sumstats_path),
                    p_threshold=cfg["step5_finemapping"]["p_threshold"],
                    window_kb=cfg["step5_finemapping"]["window_kb"],
                    merge_overlapping=cfg["step5_finemapping"][
                        "merge_overlapping"],
                    max_loci=int(bm_n),
                    output=pheno_dir / f"loci_top{int(bm_n)}.tsv",
                )
                n_capped = len(loci_capped)
                if n_capped > 0:
                    try:
                        import cupy as _cp
                        _cp.get_default_memory_pool().free_all_blocks()
                        _cp.get_default_pinned_memory_pool().free_all_blocks()
                    except Exception:  # noqa: BLE001
                        pass
                    t = time.time()
                    bm_kw = dict(common_kw)
                    bm_kw["loci"] = loci_capped
                    bm_kw["output_dir"] = pheno_dir / "finemapping_top"
                    fm_fn(**bm_kw)
                    elapsed = time.time() - t
                    per_stage_t["step5b_capped"] = elapsed
                    per_stage_t["step5b_capped_n_loci"] = n_capped
                    per_stage_t["step5b_s_per_1k_loci"] = (
                        elapsed / n_capped * 1000.0)
                    log.info(
                        f"step5b capped pass: {n_capped} loci in "
                        f"{elapsed:.1f}s "
                        f"({per_stage_t['step5b_s_per_1k_loci']:.2f} s / 1k loci)"
                    )
        elif fm_cfg.get("run", False):
            log.info("step5b requested but step5a produced no loci_df; skipping")
        else:
            log.info("step5b skipped (run=False)")

        results[pheno_name] = {
            "output_dir": str(pheno_dir),
            "timings_s": per_stage_t,
        }

    log.info(f"ultra_workflow complete in {time.time()-t_start:.1f}s "
             f"({len(phenotypes)} pheno)")
    return {
        "config": cfg,
        "results": results,
        "elapsed_s": time.time() - t_start,
        "dry_run": False,
    }


def write_example_config(path: Union[str, Path]) -> None:
    """Convenience re-export of cugen.config.write_example_config."""
    from .config import write_example_config as _w
    _w(path)
