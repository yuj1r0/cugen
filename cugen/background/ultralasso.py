"""UltraLasso as a background engine: the GPU joint LASSO + LOCO of steps 1-3.

Re-exports :func:`cugen.fit_joint_lasso` so the four engines sit side by side in
:mod:`cugen.background`. UltraLasso is the biobank-scale method (CUDA required, n in the
hundreds of thousands); at small n its lasso background overfits, the LOCO offset
absorbs the signal, and CuGenie / UltraLOCO are the calibrated choices.
"""
from ..lasso import fit_joint_lasso  # noqa: F401

__all__ = ["fit_joint_lasso"]
