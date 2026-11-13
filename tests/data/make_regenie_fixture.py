"""Write the synthetic cohort behind the REGENIE step-1 gold fixture.

    python tests/data/make_regenie_fixture.py OUTDIR
    regenie --step 1 --bed OUTDIR/regenie_fixture --phenoFile OUTDIR/regenie_fixture.pheno \
        --covarFile OUTDIR/regenie_fixture.covar --bsize 25 --force-step1 --out OUTDIR/regenie_gold

then copy regenie_fixture.{bed,bim,fam,pheno,covar} and regenie_gold_1.loco into tests/data
(see tests/data/README.md). n = 300 samples, 150 variants on chromosomes 1-3 with 2 %
missing calls, one polygenic phenotype (h2 = 0.5) and two covariates.
"""
import os
import sys

import numpy as np

from cugen._codec import BED_MAGIC, pack_bed


def main(out):
    rng = np.random.default_rng(2026)
    n, m_per_chr = 300, (60, 50, 40)
    chrom = np.concatenate([np.full(m, c + 1) for c, m in enumerate(m_per_chr)])
    m = len(chrom)
    p = rng.uniform(0.05, 0.5, m)
    G = rng.binomial(2, p[:, None], size=(m, n)).astype(np.uint8)      # counts of bim A1
    G[rng.random((m, n)) < 0.02] = 3
    X = np.where(G == 3, 0, G).astype(float)
    X = (X - X.mean(1, keepdims=True)) / (X.std(1, keepdims=True) + 1e-9)
    beta = np.zeros(m)
    cs = rng.choice(m, 25, replace=False)
    beta[cs] = rng.normal(0, 1, 25)
    gv = beta @ X
    gv = gv / gv.std() * np.sqrt(0.5)
    C = rng.normal(size=(n, 2))
    y = gv + C @ np.array([0.3, -0.2]) + rng.normal(0, np.sqrt(0.5), n)
    os.makedirs(out, exist_ok=True)
    pre = os.path.join(out, "regenie_fixture")
    with open(pre + ".bed", "wb") as fh:
        fh.write(BED_MAGIC)
        fh.write(pack_bed(G, counted="A1").tobytes())
    with open(pre + ".bim", "w") as fh:
        for i in range(m):
            fh.write(f"{chrom[i]}\tsnp{i}\t0\t{(i + 1) * 1000}\tA\tG\n")
    ids = [f"s{i}" for i in range(n)]
    with open(pre + ".fam", "w") as fh:
        for s in ids:
            fh.write(f"{s} {s} 0 0 0 -9\n")
    with open(pre + ".pheno", "w") as fh:
        fh.write("FID IID y\n")
        for s, v in zip(ids, y):
            fh.write(f"{s} {s} {v:.10g}\n")
    with open(pre + ".covar", "w") as fh:
        fh.write("FID IID c1 c2\n")
        for s, (a, b) in zip(ids, C):
            fh.write(f"{s} {s} {a:.10g} {b:.10g}\n")


if __name__ == "__main__":
    main(sys.argv[1])
