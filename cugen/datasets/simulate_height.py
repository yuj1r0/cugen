"""CLI: python -m cugen.datasets.simulate_height --out DIR [--preset n3k|n20k] [options].

Thin wrapper around ``cugen.datasets.simulate_height`` (implementation in
``cugen/datasets/_height.py``). Options left unset take the preset's value, or the
defaults (= ``n20k``) without a preset.
"""
import argparse
import json
import logging

from ._height import DEFAULT_PARAMS, PRESETS, simulate_height

# option -> (parameter, type); every option defaults to None = "preset / default value"
_OPTS = [
    ("--n", "n", int, "training samples (default 20000)"),
    ("--m", "m", int, "variants (default 1000000)"),
    ("--n-chrom", "n_chrom", int, "autosomes (default 22)"),
    ("--chrom-lengths", "chrom_lengths", str,
     "grch37 | equal | comma list of relative lengths (default grch37 up to 22 chromosomes, else equal)"),
    ("--h2", "h2", float, "SNP heritability (default 0.8)"),
    ("--n-causal", "n_causal", int, "causal variants (default 2000)"),
    ("--n-large", "n_large", int, "large-effect loci (default 10)"),
    ("--large-frac", "large_frac", float, "variance share of the large loci (default 0.10)"),
    ("--n-pops", "n_pops", int, "populations (default 1)"),
    ("--fst", "fst", float, "Balding-Nichols Fst (default 0)"),
    ("--strat-shift", "strat_shift", float, "population mean shift of the phenotype (default 0)"),
    ("--block-mean", "block_mean", float, "mean LD-block length in variants (default 150)"),
    ("--n-founders", "n_founders", int, "founder haplotypes per block (default 32)"),
    ("--maf-min", "maf_min", float, "minimum MAF (default 0.01)"),
    ("--seed", "seed", int, "master seed (default 2026)"),
    ("--formats", "formats", str, "comma list of cugen,bed (default both)"),
    ("--n-test", "n_test", int, "held-out samples, 0 = none (default 5000)"),
    ("--maf-beta", "maf_beta", str, "a,b of the target MAF Beta (default 1.0,1.5)"),
    ("--spacing-bp", "spacing_bp", float, "mean variant spacing (default 2900)"),
    ("--ld-scale-bp", "ld_scale_bp", float, "founder AR(1) correlation length (default 100000)"),
    ("--switch-rate", "switch_rate", float, "founder switch rate (default 0.01)"),
    ("--founder-conc", "founder_conc", float, "Dirichlet concentration of founder weights (default 0.5)"),
    ("--large-maf-min", "large_maf_min", float, "minimum MAF of a large locus (default 0.05)"),
    ("--age-effect", "age_effect", float, "age effect per year (default -0.01)"),
    ("--sex-effect", "sex_effect", float, "sex effect (default 0.4)"),
    ("--n-pcs", "n_pcs", int, "PCs written (default 10)"),
    ("--pc-max-snps", "pc_max_snps", int, "variants used for the PCs (default 5000)"),
]


def _parser():
    ap = argparse.ArgumentParser(
        prog="python -m cugen.datasets.simulate_height",
        description="Simulate a synthetic 'exaggerated human height' GWAS dataset "
                    "(per-chromosome .cugen + PLINK bed, phenotype, truth, config).")
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--preset", choices=sorted(PRESETS), default=None,
                    help="named parameter set (n20k = the defaults, n3k = small demo)")
    for flag, dest, typ, hlp in _OPTS:
        ap.add_argument(flag, dest=dest, type=typ, default=None, help=hlp)
    ap.add_argument("--n-workers", type=int, default=1, help="processes over chromosomes (output unchanged)")
    ap.add_argument("--no-checksums", action="store_true", help="skip the per-file sha256 in meta.json")
    ap.add_argument("--quiet", action="store_true")
    return ap


def main(argv=None):
    a = _parser().parse_args(argv)
    params = {}
    for _, dest, _, _ in _OPTS:
        v = getattr(a, dest)
        if v is None:
            continue
        if dest == "formats":
            v = tuple(f for f in v.split(",") if f)
        elif dest == "maf_beta":
            v = tuple(float(x) for x in v.split(","))
        elif dest == "chrom_lengths" and v not in ("grch37", "equal"):
            v = [float(x) for x in v.split(",")]
        params[dest] = v
    assert set(params) <= set(DEFAULT_PARAMS)
    if not a.quiet:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    meta = simulate_height(a.out, a.preset, n_workers=a.n_workers, verbose=not a.quiet,
                           checksums=not a.no_checksums, **params)
    print(json.dumps({k: meta[k] for k in ("realised", "timings_s", "sizes_bytes")}, indent=2))


if __name__ == "__main__":
    main()
