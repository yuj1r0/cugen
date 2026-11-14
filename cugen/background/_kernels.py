"""Shared kernel pass for CuGenie-K and UltraLOCO: per-chromosome sample-space Gram matrices.

    S_c = sum over variants v on chromosome c of z_v z_v'      (n x n)
    N_c = sum over variants of m_v m_v' (valid-call masks)      (GCTA pairwise marker counts)
    M_c = number of variants used on chromosome c

``scaling="regenie"`` (CuGenie-K): mean-impute, project out the covariates, ||z||^2 = n - ncov.
``scaling="gcta"`` (UltraLOCO): (g - 2p)/sqrt(2p(1-p)), missing -> 0, plus N_c when
``pairwise_count=True``.

Memory modes: ``low_memory=False`` keeps every S_c (C n^2 words, plus N_c for pairwise
counts); ``low_memory=True`` keeps only the totals and recomputes S_c (N_c) per chromosome on
demand with a second read of that chromosome. Either way ``KernelSet.loco(c)`` returns
K_-c = (S - S_c) / (M - M_c) (or / (N - N_c) elementwise).
"""
from __future__ import annotations

import time

import numpy as np

from ._prep import standardize_gcta, standardize_regenie


class KernelSet:
    def __init__(self, geno, rows, dev, scaling, Q=None, ncov=0, pairwise_count=False,
                 low_memory=False, dtype="float64", block_size=4096, prefetch=2, verbose=True):
        self.geno, self.rows, self.dev = geno, rows, dev
        self.scaling = scaling
        self.Q, self.ncov = Q, ncov
        self.pairwise = bool(pairwise_count)
        self.low_memory = bool(low_memory)
        self.dtype = np.dtype(dtype)
        self.block_size = int(block_size)
        self.prefetch = prefetch
        self.verbose = verbose
        self.ranges = {c: (s, e) for c, s, e in geno.chrom_ranges(rows)}
        self.chroms = sorted(self.ranges)
        self.S, self.N = {}, {}
        self._dropped = set()
        self.M = {}
        self.n_lowvar = 0
        self.timing = {"kernel_pass_s": 0.0, "kernel_recompute_s": 0.0}

    # one chromosome (or a list) -> accumulate into the given dicts
    def _accumulate(self, chroms, per_chrom):
        dev, xp, n = self.dev, self.dev.xp, self.geno.n
        acc_dt = xp.float64
        blocks = []
        for c in chroms:
            s0, e0 = self.ranges[c]
            for s in range(s0, e0, self.block_size):
                blocks.append((c, s, min(s + self.block_size, e0)))
        S, N, M = {}, {}, {}
        Stot = Ntot = None
        nlow = 0
        for c, s, e, counts in self.geno.iter_blocks(self.rows, blocks, dev, self.prefetch):
            if self.scaling == "regenie":
                Z, ok = standardize_regenie(counts, self.Q, self.ncov, xp, dtype=xp.float64)
                mk = None
            else:
                Z, mk, ok = standardize_gcta(counts, xp, dtype=xp.dtype(self.dtype))
            del counts
            okh = dev.asnumpy(ok)
            nlow += int((~okh).sum())
            if not okh.all():
                sel = dev.asarray(np.flatnonzero(okh))
                Z = Z[sel]
                if mk is not None:
                    mk = mk[sel]
            Z = Z.astype(xp.dtype(self.dtype), copy=False)
            contrib = (Z.T @ Z).astype(acc_dt, copy=False)
            if per_chrom:
                if c not in S:
                    S[c] = xp.zeros((n, n), dtype=acc_dt)
                S[c] += contrib
            else:
                Stot = contrib if Stot is None else Stot + contrib
            del contrib
            if self.pairwise:
                cm = (mk.T @ mk).astype(acc_dt, copy=False)
                if per_chrom:
                    if c not in N:
                        N[c] = xp.zeros((n, n), dtype=acc_dt)
                    N[c] += cm
                else:
                    Ntot = cm if Ntot is None else Ntot + cm
                del cm
            M[c] = M.get(c, 0) + int(okh.sum())
        dev.sync()
        return S, N, M, Stot, Ntot, nlow

    def build(self):
        t = time.time()
        if self.low_memory:
            _, _, M, Stot, Ntot, nlow = self._accumulate(self.chroms, per_chrom=False)
            self.Stot, self.Ntot = Stot, Ntot
        else:
            S, N, M, _, _, nlow = self._accumulate(self.chroms, per_chrom=True)
            self.S, self.N = S, N
            self.Stot = sum(S.values())
            self.Ntot = sum(N.values()) if self.pairwise else None
        self.M = {c: M.get(c, 0) for c in self.chroms}
        self.Mtot = int(sum(self.M.values()))
        self.n_lowvar = nlow
        self.timing["kernel_pass_s"] += time.time() - t
        return self

    def chrom_parts(self, c):
        """(S_c, N_c or None) for chromosome c (recomputed in low_memory mode or after drop(c))."""
        if not self.low_memory and c not in self._dropped:
            return self.S.get(c), (self.N.get(c) if self.pairwise else None)
        t = time.time()
        S, N, _, _, _, _ = self._accumulate([c], per_chrom=True)
        self.timing["kernel_recompute_s"] += time.time() - t
        return S.get(c), (N.get(c) if self.pairwise else None)

    def loco(self, c):
        """K_-c, symmetrised."""
        Sc, Nc = self.chrom_parts(c)
        if Sc is None:
            Sc = 0
        if self.pairwise:
            K = (self.Stot - Sc) / (self.Ntot - (Nc if Nc is not None else 0))
        else:
            K = (self.Stot - Sc) / max(self.Mtot - self.M.get(c, 0), 1)
        return (K + K.T) * 0.5

    def drop(self, c):
        """Free S_c / N_c once chromosome c is done (full mode); a later loco(c) recomputes them."""
        if c in self.S or c in self.N:
            self._dropped.add(c)
        self.S.pop(c, None)
        self.N.pop(c, None)


def kernel_bytes(n, n_chrom, pairwise, low_memory, work=6, itemsize=8):
    """Peak bytes of the kernel state + the per-chromosome eigh working set."""
    per = 2 if pairwise else 1
    keep = (per * (1 if low_memory else n_chrom) + per) * n * n * itemsize   # parts + totals
    if low_memory:
        keep += per * n * n * itemsize                                       # recomputed S_c/N_c
    return int(keep + work * n * n * itemsize)
