"""cugen.ultralasso — the ultraLasso GWAS method (steps 1-5b in one namespace).

scikit-learn-style namespace::

    from cugen import ultralasso

    # Steps 1+2 (per-chr SNP screening)
    cands  = ultralasso.screen(chrom=22, cohort_npz=..., cugen_path=...)
    # Step 3 (joint LASSO + LOCO predictions)
    fit    = ultralasso.fit(cands, cohort_npz=..., cugen_dir=...)
    # Step 4 (GWAS)
    ss     = ultralasso.gwas(cugen_dir=..., cohort_npz=..., loco_predictions=...)
    # Step 5a (locus definition)
    loci   = ultralasso.loci(ss)
    # Step 5b Tier 1 (UltraSuSiE genome-wide fine-mapping)
    fm     = ultralasso.susie(loci, loco_predictions=..., cohort_npz=...,
                               cugen_dir=..., annotation=..., output_dir=...)
    # Step 5b Tier 2 (UltraMAP dense-LD adaptive-LASSO + consensus)
    fm2    = ultralasso.map_(loci, ..., adaptive_weight_method='z')

    # Or end-to-end:
    result = ultralasso.run(cohort_npz=..., cugen_dir=...)

    # Polygenic score (PRS): build train/val/test splits + subset cugens, then
    # sweep step-3 alpha on validation, refit the best on train+val, score test:
    ultralasso.build_prs_splits(ref_npz=..., master_phe=..., source_cugen_dir=...,
                                phenotypes=[("INI50", False)], splits={...},
                                out_dir=..., scratch_dir=...)
    res = ultralasso.prs("INI50", splits_dir=..., cohorts_dir=..., score_cugen_dir=...,
                         train_cugen_dir=..., trainval_cugen_dir=..., ref_npz=...,
                         master_phe=..., step2_alpha=2e-4, out_dir=...)

The canonical computation lives in the sibling flat modules
(``cugen.screen``, ``cugen.lasso``, ``cugen.assoc``,
``cugen.loci``, ``cugen.finemapping``); this subpackage simply
provides the manuscript-aligned namespace so the whole UltraLasso pipeline
— including UltraSuSiE Tier 1 fine-mapping and UltraMAP Tier 2 — is a
single import away and JSON-config-drivable from
:func:`cugen.ultra_workflow`.

The byte-exact production ports (validated session 31, jobs 26412730 +
26412732 vs ``results_pipelined_fwb_ini50_fixI/``) are unchanged.
Fine-mapping inherits production correctness via the
:mod:`step5b_finemapping` shim — see ``ultra/cugen/finemapping.py``.

Naming caveat: Python keyword ``map`` is reserved, so UltraMAP is exposed
as ``ultralasso.map_`` (trailing underscore). The susie / map_ split matches
the manuscript's Tier 1 / Tier 2 framing.

See the manuscript §2.1 "The ultraLasso GWAS framework" for the method
description.
"""

from ..assoc import gwas, ultralasso_gwas as run
from ..finemapping import ultramap as map_, ultrasusie as susie
from ..lasso import fit_joint_lasso as fit
from ..loci import define_loci as loci
from ..prs import (
    build_prs_splits, prs,
    prs_load_phenotype, prs_residualize, prs_screen, prs_fit_weights, prs_score,
    prs_alpha_sweep, select_best_alpha, prs_r2, prs_auc,
)
from ..screen import screen_chromosome as screen

__all__ = ["screen", "fit", "gwas", "run", "loci", "susie", "map_",
           "prs", "build_prs_splits",
           "prs_load_phenotype", "prs_residualize", "prs_screen", "prs_fit_weights",
           "prs_score", "prs_alpha_sweep", "select_best_alpha", "prs_r2", "prs_auc"]
