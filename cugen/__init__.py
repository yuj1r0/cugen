"""cugen — GPU-accelerated UltraLasso genomics API.

Canonical entry points::

    import cugen as cg

    cg.ultra_workflow("config.json")              # full pipeline from JSON

    # Or step-by-step:
    cohort = cg.prepare_cohort(pheno, covariates, master_phe)
    cands  = cg.screen_chromosome(chrom, cohort, cugen_path)
    actives, loco = cg.fit_joint_lasso(cands, cohort, cugens, covars)
    sumstats = cg.gwas(cugen_dir, loco, cohort)
    loci   = cg.define_loci(sumstats)
    fm     = cg.ultralasso.susie(loci, loco, cugens,
                                  annotation=..., output_dir=...)

scikit-learn-style namespace (manuscript-aligned) — UltraSuSiE and
UltraMAP are step 5b of the *one* UltraLasso pipeline, so they live under
``cg.ultralasso``, not at the top level::

    from cugen import ultralasso

    cands  = ultralasso.screen(...)        # step 1+2
    fit    = ultralasso.fit(cands, ...)    # step 3 joint LASSO + LOCO
    ss     = ultralasso.gwas(fit, ...)     # step 4
    loci   = ultralasso.loci(ss)           # step 5a
    fm     = ultralasso.susie(loci, ...)   # step 5b Tier 1 (UltraSuSiE)
    fm2    = ultralasso.map_(loci, ...)    # step 5b Tier 2 (UltraMAP)

PLINK / Hail aliases are wired in too (cg.keep, cg.glm, cg.freq, …).
"""

__version__ = "0.1.7"


def check_gpu():
    """Return dict with GPU availability + memory info, or {'available': False}."""
    try:
        import cupy as cp
        dev = cp.cuda.Device()
        free, total = dev.mem_info
        return {
            "available": True,
            "device_id": dev.id,
            "name": cp.cuda.runtime.getDeviceProperties(dev.id)["name"].decode(),
            "memory_free_gb": free / 1e9,
            "memory_total_gb": total / 1e9,
        }
    except Exception as e:
        return {"available": False, "error": str(e)}


def set_device(device_id: int):
    """Select CUDA device for subsequent CuPy ops."""
    import cupy as cp
    cp.cuda.Device(device_id).use()
    return device_id


# --- canonical API ---
from .io import CugenReader, read_cugen, read_cugen_header
from .prep_cohort import prepare_cohort
from .screen import screen_chromosome
from .lasso import fit_joint_lasso
from .assoc import gwas, ultralasso_gwas
from .loci import define_loci
from .workflow import ultra_workflow
from .freq import frequency

# v0.1.2: real subset implementation (sample-axis filter + stat recompute).
from .subset import (
    filter_cols,
    filter_rows,
    subset_cugen_dir,
    subset_cugen_file,
)

# session 51: in-place stats-block repair (recompute mu_x/sxx/maf from the
# file's own genotypes; fixes corrupt precomputed stats without recomputing
# inside the association). cg.repair dispatches on file-vs-dir.
from .repair import repair, repair_cugen_dir, repair_cugen_file

# v0.1.3: real implementations for QC + PRS scoring. ld/popstruct still stubs.
from .qc import sample_qc, variant_qc
from .score import score
from .ld import ld_clump, ld_matrix
from .popstruct import grm, king, pca, pc_project

# --- alias wiring (PLINK + Hail) ---
from .aliases import (  # noqa: F401
    clump,
    exclude,
    extract,
    freq,
    glm,
    hwe_normalized_pca,
    keep,
    linear_regression_rows,
    logistic_regression_rows,
    make_grm,
    make_king,
    r2,
    realized_relationship_matrix,
    remove,
    set_toolkit,
)

# --- method subpackage (sklearn-style namespace; manuscript-aligned) ---
# UltraSuSiE (Tier 1) + UltraMAP (Tier 2) live INSIDE ultralasso as
# ``ultralasso.susie`` / ``ultralasso.map_``, not as top-level siblings,
# because they are step 5b of the ONE UltraLasso pipeline.
from . import ultralasso  # noqa: F401

# --- stub modules (v0.2 roadmap) ---
from . import ld, plot, popstruct, qc, score, subset  # noqa: F401

__all__ = [
    # diagnostics
    "check_gpu",
    "set_device",
    # canonical
    "CugenReader",
    "read_cugen",
    "read_cugen_header",
    "prepare_cohort",
    "screen_chromosome",
    "fit_joint_lasso",
    "gwas",
    "ultralasso_gwas",
    "define_loci",
    "ultra_workflow",
    "frequency",
    # subsetting (v0.1.2)
    "subset_cugen_file",
    "subset_cugen_dir",
    "filter_cols",
    "filter_rows",
    # stats-block repair (session 51)
    "repair",
    "repair_cugen_file",
    "repair_cugen_dir",
    # QC + PRS (v0.1.3)
    "variant_qc",
    "sample_qc",
    "score",
    # method subpackage (sklearn-style; contains susie + map_ for step 5b)
    "ultralasso",
    # PLINK aliases
    "keep",
    "extract",
    "remove",
    "exclude",
    "freq",
    "glm",
    "clump",
    "r2",
    "make_king",
    "make_grm",
    # Hail aliases
    "linear_regression_rows",
    "logistic_regression_rows",
    "hwe_normalized_pca",
    "realized_relationship_matrix",
    "set_toolkit",
]
