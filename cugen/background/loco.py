"""LocoPredictions: the LOCO background every engine in :mod:`cugen.background` emits.

It reads and writes CuGen's ``loco_predictions.npz`` contract, so step 4 consumes these
offsets whichever engine produced them:

    predictions : (n, C) float32; column j = prediction of y leaving chromosome
                  ``chromosomes[j]`` out (covariate fit included:
                  predictions = yhat_cov + scale_y * offset)
    fid_order   : (n,) sample IDs (unicode, never truncated)
    y_original  : (n,) float32 phenotype on its original scale
    chromosomes : (C,) int64 chromosome of each column. Files without it are read as
                  chromosomes 1..C, which is also what the engines write by default
                  (1..max chromosome of the panel), so column c-1 is chromosome c.

Use :meth:`LocoPredictions.column` rather than ``c - 1`` when the chromosome set may not be
1..C (e.g. ``chromosomes=[2, 5, 7]``).
"""
from __future__ import annotations

import logging
import os

import numpy as np

logger = logging.getLogger("cugen.background.loco")


def _as_chromosomes(chromosomes, C):
    if chromosomes is None:
        return np.arange(1, C + 1, dtype=np.int64)
    ch = np.asarray(list(chromosomes), dtype=np.int64)
    if ch.ndim != 1 or len(ch) != C:
        raise ValueError(f"chromosomes has {ch.size} entries but predictions have {C} columns")
    if len(ch) > 1 and np.any(np.diff(ch) <= 0):
        raise ValueError("chromosomes must be strictly increasing")
    return ch


class LocoPredictions:
    """A LOCO background: predictions (n, C) or (n, C, P) on the phenotype scale.

    One column per chromosome in ``chromosomes`` (default 1..C) and one slice per
    phenotype; ``offsets`` / ``scale_y`` / ``yhat_cov`` keep the standardised-scale pieces
    when an engine built the object. ``method`` and ``meta`` describe how it was fitted.
    """

    def __init__(self, predictions, y_original, fid_order=None, phenotypes=None, method=None,
                 meta=None, offsets=None, scale_y=None, yhat_cov=None, chromosomes=None):
        p = np.asarray(predictions, dtype=np.float64)
        y = np.asarray(y_original, dtype=np.float64)
        if p.ndim == 2:
            p = p[:, :, None]
        if y.ndim == 1:
            y = y[:, None]
        if p.shape[0] != y.shape[0] or p.shape[2] != y.shape[1]:
            raise ValueError(f"predictions {p.shape} and y_original {y.shape} do not align")
        self.predictions = p
        self.y_original = y
        n = p.shape[0]
        self.fid_order = (np.arange(n).astype(str) if fid_order is None else np.asarray(fid_order).astype(str))
        if len(self.fid_order) != n:
            raise ValueError(f"fid_order has {len(self.fid_order)} IDs, predictions have {n} rows")
        self.chromosomes = _as_chromosomes(chromosomes, p.shape[1])
        self.phenotypes = list(phenotypes) if phenotypes is not None else (
            ["y"] if p.shape[2] == 1 else [f"y{k}" for k in range(p.shape[2])])
        self.method = method
        self.meta = dict(meta or {})
        self.offsets = offsets
        self.scale_y = scale_y
        self.yhat_cov = yhat_cov

    @property
    def n(self):
        return self.predictions.shape[0]

    @property
    def n_chrom(self):
        return self.predictions.shape[1]

    def __len__(self):
        return self.predictions.shape[2]

    def column(self, chrom):
        """Column index of chromosome ``chrom`` (KeyError if the background lacks it)."""
        j = np.flatnonzero(self.chromosomes == int(chrom))
        if not len(j):
            raise KeyError(f"chromosome {chrom} is not in this LOCO background "
                           f"(chromosomes {self.chromosomes.tolist()})")
        return int(j[0])

    def __getitem__(self, k):
        if isinstance(k, str):
            k = self.phenotypes.index(k)
        yc = self.yhat_cov
        return LocoPredictions(self.predictions[:, :, k], self.y_original[:, k], self.fid_order,
                               [self.phenotypes[k]], self.method, self.meta,
                               None if self.offsets is None else self.offsets[:, :, k],
                               None if self.scale_y is None else np.atleast_1d(self.scale_y)[k:k + 1],
                               None if yc is None else (yc[..., k] if np.ndim(yc) >= 2 else yc[k]),
                               self.chromosomes)

    def items(self):
        for k, nm in enumerate(self.phenotypes):
            yield nm, self[k]

    def prediction(self, chrom, k=0):
        """Prediction of phenotype ``k`` leaving chromosome ``chrom`` out."""
        return self.predictions[:, self.column(chrom), k]

    def residual(self, chrom, k=0):
        """y_original - prediction leaving chromosome `chrom` out."""
        return self.y_original[:, k] - self.prediction(chrom, k)

    def save(self, path):
        """One phenotype: write ``path`` (if it ends in .npz) or ``path/loco_predictions.npz``.
        Several: ``path/<phenotype>/loco_predictions.npz``. Returns the written paths."""
        contiguous = np.array_equal(self.chromosomes, np.arange(1, self.n_chrom + 1))
        if not contiguous:
            logger.warning("saving LOCO predictions for chromosomes %s: consumers that index columns as "
                           "chromosome - 1 must use the stored 'chromosomes' field",
                           self.chromosomes.tolist())
        out = []
        for k, nm in enumerate(self.phenotypes):
            if len(self) == 1:
                f = path if str(path).endswith(".npz") else os.path.join(path, "loco_predictions.npz")
            else:
                f = os.path.join(path, nm, "loco_predictions.npz")
            os.makedirs(os.path.dirname(os.path.abspath(f)), exist_ok=True)
            np.savez_compressed(f, predictions=self.predictions[:, :, k].astype(np.float32),
                                fid_order=self.fid_order.astype(str),
                                y_original=self.y_original[:, k].astype(np.float32),
                                chromosomes=self.chromosomes.astype(np.int64))
            out.append(str(f))
        return out

    @classmethod
    def load(cls, path, phenotype=None):
        """Read a ``loco_predictions.npz`` (path to the file or to its directory)."""
        f = path if str(path).endswith(".npz") else os.path.join(path, "loco_predictions.npz")
        with np.load(f, allow_pickle=False) as d:
            pred = d["predictions"]
            yo = d["y_original"]
            fid = d["fid_order"] if "fid_order" in d.files else None
            ch = d["chromosomes"] if "chromosomes" in d.files else None
        return cls(pred, yo, fid,
                   [phenotype or os.path.basename(os.path.dirname(os.path.abspath(f))) or "y"],
                   method="loaded", meta={"path": str(f)}, chromosomes=ch)

    def __repr__(self):
        ch = self.chromosomes
        cs = (f"1..{ch[-1]}" if np.array_equal(ch, np.arange(1, len(ch) + 1)) and len(ch) else ch.tolist())
        return (f"LocoPredictions(method={self.method!r}, n={self.n}, chromosomes={cs}, "
                f"phenotypes={self.phenotypes})")


def build(offsets, scale_y, yhat_cov, y_original, fid_order, phenotypes, method, meta, chromosomes):
    """offsets (n, C, P) on the standardised scale -> LocoPredictions on the phenotype scale."""
    pred = yhat_cov[:, None, :] + scale_y[None, None, :] * offsets
    return LocoPredictions(pred, y_original, fid_order, phenotypes, method, meta, offsets=offsets,
                           scale_y=np.asarray(scale_y), yhat_cov=yhat_cov, chromosomes=chromosomes)


def output_chromosomes(present, chromosomes=None):
    """LOCO columns: ``chromosomes`` (sorted, unique) when given, else 1..max(present).

    The default keeps column c-1 = chromosome c, which every existing consumer assumes;
    chromosomes of the range without panel variants get the full-panel prediction
    (CuGenie masked) or the covariate fit (offset 0)."""
    if chromosomes is not None:
        ch = sorted({int(c) for c in chromosomes})
        missing = sorted(set(present) - set(ch))
        if missing:
            raise ValueError(f"panel has chromosomes {missing} outside chromosomes={ch}")
        return ch
    if not present:
        raise ValueError("no variants selected")
    return list(range(1, int(max(present)) + 1))
