# LD test fixtures

`ld_fixture.npy` -- (12 variants, 60 samples) uint8 dosages, 3 = missing.
Variant 11 is monomorphic; variants 4 and 9 carry missing calls; ALT frequency
straddles 0.5 so the "alt" and "major" sign conventions genuinely diverge.

Golden files produced with **PLINK v2.0.0-a.7.1 M1 (4 May 2026)**:

```
plink2 --vcf ld_fixture.vcf --make-pgen --out t
plink2 --pfile t --r-unphased allow-ambiguous-allele \
  cols=chrom,pos,id,ref,alt,maj,nonmaj,freq \
  --ld-window 999999 --ld-window-kb 99999 --ld-window-r2 0 --out u
plink2 --pfile t --r-phased allow-ambiguous-allele \
  cols=chrom,pos,id,ref,alt,maj,nonmaj,freq,d,dprime \
  --ld-window 999999 --ld-window-kb 99999 --ld-window-r2 0 --out p
```

`--ld-window-r2 0` is mandatory: plink2 defaults to 0.2 and would silently drop
most pairs. `--ld-window` has NO default in plink2 (it was 10 in plink 1.9), so
it is set explicitly. `allow-ambiguous-allele` is a MODIFIER on --r*, not a
standalone flag.

The .vcor files carry ~6 significant figures, so compare with rtol=1e-6.

## Clumping fixtures

`clump_fixture.npz` -- `G` (150 variants, 250 samples) uint8 dosages laid out
in LD blocks of 4-9 variants, plus `POS` (sorted, spread over 1.5 Mb so the
250 kb window bites) and `P`. `clump_sumstats.tsv` is the same p-values in
plink report form.

`clump_gold.clumps` and `clump_gold_overlap.clumps` are plink2 v2.0.0-a.7.1
output, the second with `--clump-allow-overlap`. Regenerate with:

```bash
# rebuild the VCF from the npz first (see tests/test_ld_clump.py::_clump_fixture)
plink2 --vcf cl.vcf --make-pgen --out cl
plink2 --pfile cl --clump clump_sumstats.tsv --clump-unphased \
    --clump-p1 1e-4 --clump-p2 0.01 --clump-r2 0.5 --clump-kb 250 --out gold
plink2 --pfile cl --clump clump_sumstats.tsv --clump-unphased \
    --clump-allow-overlap --clump-p1 1e-4 --clump-p2 0.01 --clump-r2 0.5 \
    --clump-kb 250 --out gold_overlap
```

`--clump-unphased` is mandatory: plink2's default `--clump` uses PHASED r^2 and
.cugen discards phase, so the unphased form is the only one we can match. Same
caveat as D/D'.

The fixture is chosen so the golden actually exercises the rules -- it contains
clumps whose `TOTAL` exceeds their `SP2` count (members above p2) and a
non-zero `NONSIG` bin. `test_golden_exercises_the_asymmetry_it_is_meant_to`
asserts that, because a parity test that cannot fail is worthless: an earlier
implementation gated membership on p2, produced the right index variants with
systematically low `TOTAL`, and would have passed a golden without them.

## REGENIE step-1 fixture

`regenie_fixture.{bed,bim,fam}` -- 300 samples x 150 variants on chromosomes 1-3 (60/50/40),
2 % missing calls, written by `make_regenie_fixture.py` (seed 2026). `regenie_fixture.pheno`
holds one polygenic phenotype (h2 = 0.5) and `regenie_fixture.covar` two covariates.
`regenie_gold_1.loco` is the LOCO output of **REGENIE v2.2.4** step 1 with its defaults
(5 contiguous folds, h2 grid 0.01/0.25/0.5/0.75/0.99 at both levels):

```bash
python tests/data/make_regenie_fixture.py OUT
regenie --step 1 --bed OUT/regenie_fixture --phenoFile OUT/regenie_fixture.pheno \
    --covarFile OUT/regenie_fixture.covar --bsize 25 --force-step1 --out OUT/regenie_gold
```

`--force-step1` is needed because the panel is tiny. The .loco lists samples in REGENIE's own
order (not .fam order) and has rows for chromosomes 1-23, so match columns by ID and rows by
chromosome. Values are written with ~6 significant digits: `tests/test_regenie_gold.py`
compares CuGenie's masked LOCO offsets with atol 1e-5 (observed max |diff| 4.6e-6, r > 0.9999999999)
and the level-1 R^2 / MSE with the digits REGENIE prints in its log.

