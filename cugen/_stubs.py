"""Helper for stub modules planned for v0.2+."""


def _stub(name: str):
    raise NotImplementedError(
        f"cugen.{name} is planned for v0.2 — see README roadmap. "
        f"For v0.1 the implemented modules are: io, genotypes, prep_cohort, screen, "
        f"lasso, assoc, background, loci, finemapping, workflow, freq, qc, score, ld, "
        f"impute."
    )
