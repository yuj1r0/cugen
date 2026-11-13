"""cugen.datasets - demo and benchmark datasets (code only; numpy + standard library).

Synthetic data (generated locally, deterministic for a seed)::

    simulate_height(out_dir, preset="n3k")     # or "n20k" (the defaults), or any parameters
    workflow_config(data_dir, output_dir)      # ready ``cg.ultra_workflow`` config
    evaluate_gwas(sumstats, data_dir)          # score a GWAS against the simulated truth
    evaluate_finemapping(fm_dir, data_dir)     # score UltraSuSiE credible sets
    compare_sumstats(a, b)                     # r(z) / slope / loci overlap of two runs
    background_r2(run_dir, data_dir)           # in-sample vs held-out R2 of the step-3 model

Published data (checksum-verified download into a cache directory)::

    list_datasets(); info(name)
    fetch(name, dest=None, parts=None)         # -> dataset directory
    verify(path, name)                         # check files already on disk

CLI: ``python -m cugen.datasets.simulate_height --out DIR --preset n3k``.
"""
from ._height import (  # noqa: F401
    DEFAULT_PARAMS,
    GRCH37_AUTOSOME_BP,
    PRESETS,
    STANDARD_COVARIATES,
    regenerate_command,
    simulate_height,
    workflow_config,
    write_workflow_config,
)
from .evaluate import (  # noqa: F401
    background_r2,
    clump_loci,
    compare_sumstats,
    evaluate_finemapping,
    evaluate_gwas,
    load_regenie,
    load_sumstats,
)
from ._fetch import (  # noqa: F401
    ChecksumError,
    DatasetError,
    DatasetNotHostedError,
    ManualDownloadRequired,
    data_home,
    fetch,
    info,
    list_datasets,
    verify,
)

__all__ = ["simulate_height", "workflow_config", "write_workflow_config", "regenerate_command",
           "evaluate_gwas", "evaluate_finemapping", "compare_sumstats", "background_r2", "clump_loci",
           "load_sumstats", "load_regenie", "list_datasets", "info", "fetch", "verify", "data_home",
           "DatasetError", "DatasetNotHostedError", "ManualDownloadRequired", "ChecksumError",
           "PRESETS", "DEFAULT_PARAMS", "GRCH37_AUTOSOME_BP", "STANDARD_COVARIATES"]
