"""Frequency utilities — cugen header is the source of truth for MAF / sxx."""

from pathlib import Path

from .io import CugenReader


def frequency(cugen_path):
    """Read precomputed (mu_x, sxx, maf) arrays from a cugen header.

    Returns a dict with arrays of length n_variants — no genotype I/O.
    """
    p = Path(cugen_path)
    with CugenReader(str(p)) as r:
        mu_x, sxx, maf = r.get_stats(0, r.n_variants)
        return {
            "n_variants": r.n_variants,
            "n_samples": r.n_samples,
            "mu_x": mu_x,
            "sxx": sxx,
            "maf": maf,
        }
