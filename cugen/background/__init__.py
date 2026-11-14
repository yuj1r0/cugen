"""``cugen.background``: LOCO background engines, on CPU or GPU.

In CuGen the *background* is the polygenic LOCO model that step 4 tests each variant
against. Every engine here emits the same :class:`LocoPredictions` contract
(``loco_predictions.npz``), so any of them can feed step 4:

* :func:`fit_joint_lasso` (UltraLasso, :mod:`cugen.background.ultralasso`): the GPU joint
  LASSO + LOCO refits of steps 1-3; the biobank-scale default.
* :func:`fit_cugenie`: CuGenie, REGENIE-exact step 1 (block ridge + stacking, masked LOCO).
* :func:`fit_cugenie_k`: CuGenie-K, whole-genome kernel ridge with exact leave-one-out CV.
* :func:`fit_ultraloco` + :func:`ultraloco_gwas`: UltraLOCO, the GLS-LOCO mixed model
  (GCTA-MLMA-LOCO equivalent); a different test, so it has its own association.

CuGenie, CuGenie-K and UltraLOCO are one implementation for numpy and cupy
(``device="auto"|"cpu"|"cuda"|"cuda:N"``). Each takes ``estimate_only=True`` (peak memory
and FLOPs, nothing computed) and raises :class:`cugen.CugenResourceError` with
alternatives before allocating when a call will not fit.

:func:`gwas_blocked` is the blocked linear step 4 (numpy or cupy, .bed or cugen input,
several backgrounds in one pass). It is public here for now; ``cugen.gwas`` will
dispatch to it as ``engine="blocked"``.

Recipe (CPU, pruned step-1 panel)::

    import cugen as cg
    geno = cg.open_genotypes("data/cohort")                  # PLINK prefix; counts bim A1
    loco = cg.fit_cugenie(geno, y, covariates, snps="prune.prune.in")
    ss   = cg.background.gwas_blocked(geno, loco)
    info, loci = cg.summarize(ss)
"""
from .._device import CugenResourceError  # noqa: F401
from .._assoc_blocked import estimate_step4, gwas_blocked, write_sumstats  # noqa: F401
from . import ultralasso  # noqa: F401
from .cugenie import estimate_cugenie, fit_cugenie  # noqa: F401
from .cugenie_k import estimate_cugenie_k, fit_cugenie_k  # noqa: F401
from .loco import LocoPredictions  # noqa: F401
from .ultralasso import fit_joint_lasso  # noqa: F401
from .ultraloco import UltraLocoModel, estimate_ultraloco, fit_ultraloco, reml, ultraloco_gwas  # noqa: F401

__all__ = ["LocoPredictions", "fit_joint_lasso", "fit_cugenie", "fit_cugenie_k", "fit_ultraloco",
           "ultraloco_gwas", "UltraLocoModel", "reml", "gwas_blocked", "write_sumstats",
           "estimate_cugenie", "estimate_cugenie_k", "estimate_ultraloco", "estimate_step4",
           "CugenResourceError", "ultralasso"]
