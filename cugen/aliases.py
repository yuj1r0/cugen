"""PLINK and Hail style aliases mapped onto cugen's canonical API.

Wired into cugen.__init__ so users can write either PLINK-style
(`cg.keep`, `cg.glm`, `cg.freq`) or Hail-style (`cg.linear_regression_rows`)
or cugen-canonical (`cg.filter_cols`, `cg.gwas`, `cg.frequency`).
"""

from .assoc import gwas
from .freq import frequency
from .io import read_cugen, read_cugen_header
from .ld import ld_clump, ld_matrix
from .popstruct import grm, king, pca, pc_project
from .qc import sample_qc, variant_qc
from .score import score
from .subset import filter_cols, filter_rows

# -------- PLINK-style --------
keep = filter_cols
extract = filter_rows


def remove(mt, **kw):
    return filter_cols(mt, invert=True, **kw)


def exclude(mt, **kw):
    return filter_rows(mt, invert=True, **kw)


freq = frequency
glm = gwas
clump = ld_clump
r2 = ld_matrix
make_king = king
make_grm = grm

# -------- Hail-style --------
linear_regression_rows = gwas


def logistic_regression_rows(mt, **kw):
    return gwas(mt, family="logistic", **kw)


realized_relationship_matrix = grm
hwe_normalized_pca = pca

# -------- toolkit selector (for future name conflicts) --------
_toolkit = "cugen"


def set_toolkit(name: str):
    global _toolkit
    _toolkit = name
