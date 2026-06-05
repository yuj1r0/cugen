"""Helper for stub modules planned for v0.2+."""


def _stub(name: str):
    raise NotImplementedError(
        f"cugen.{name} is planned for v0.2 — see README roadmap. "
        f"For v0.1 the implemented modules are: io, prep_cohort, screen, lasso, "
        f"assoc, loci, finemapping, workflow, freq."
    )
