"""cugen.cli — `cugen` console-script entry point.

Behaviour:
  cugen config.json                run ultra_workflow on the JSON
  cugen info chr22.cugen           print file metadata
  cugen freq chr22.cugen --out X   dump per-variant (mu_x, sxx, maf)
  cugen prep --pheno height ...     build a residualised cohort NPZ
  cugen workflow --example out.json   write a fully-populated example JSON
  cugen gwas --cugen-dir … --pheno …  shortcut: ultralasso_gwas one-shot
"""

import argparse
import json
import sys
from pathlib import Path


def _add_prep(sp):
    p = sp.add_parser("prep", help="build a residualised cohort NPZ")
    p.add_argument("--pheno", required=True)
    p.add_argument("--cohort", default="unified",
                   choices=["unified", "full_wb"])
    p.add_argument("--out", required=True)
    p.add_argument("--covars", default=None,
                   help="comma-separated covariate names")
    p.add_argument("--master-phe", default=None)
    return p


def _add_info(sp):
    p = sp.add_parser("info", help="print .cugen file metadata")
    p.add_argument("cugen_file")
    return p


def _add_freq(sp):
    p = sp.add_parser("freq", help="dump per-variant (mu_x, sxx, maf) from cugen header")
    p.add_argument("cugen_file")
    p.add_argument("--out", required=False,
                   help="write TSV here; default = stdout")
    return p


def _add_workflow(sp):
    p = sp.add_parser("workflow", help="ultra_workflow utilities")
    p.add_argument("--example", metavar="PATH",
                   help="write a fully-populated example JSON config")
    p.add_argument("--validate", metavar="JSON",
                   help="validate a JSON config without executing")
    p.add_argument("--dry-run", metavar="JSON",
                   help="resolve config + plan, but skip execution")
    return p


def _add_gwas(sp):
    p = sp.add_parser("gwas", help="run ultralasso_gwas (Steps 1-3+4) one-shot")
    p.add_argument("--cugen-dir", required=True)
    p.add_argument("--pheno", required=True)
    p.add_argument("--pheno-type", default="continuous",
                   choices=["continuous", "binary"])
    p.add_argument("--imputed-dir", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--no-finemapping", action="store_true")
    p.add_argument("--covariates", nargs="+", default=None)
    return p


def _add_qc(sp):
    p = sp.add_parser("qc", help="(v0.2 roadmap)")
    p.add_argument("--cugen-dir", required=True)
    p.add_argument("--maf", type=float, default=None)
    p.add_argument("--geno", type=float, default=None)
    p.add_argument("--hwe", type=float, default=None)
    p.add_argument("--mind", type=float, default=None)
    p.add_argument("--out", required=True)
    return p


def _add_score(sp):
    p = sp.add_parser("score", help="(v0.2 roadmap)")
    p.add_argument("--cugen-dir", required=True)
    p.add_argument("--weights", required=True)
    p.add_argument("--out", required=True)
    return p


def _run_prep(args):
    from .prep_cohort import prepare_cohort
    covars = args.covars.split(",") if args.covars else None
    kw = dict(phenotype=args.pheno, cohort=args.cohort,
              output=args.out, covariates=covars)
    if args.master_phe:
        kw["master_phe"] = args.master_phe
    prepare_cohort(**kw)


def _run_info(args):
    from .io import read_cugen_header
    info = read_cugen_header(args.cugen_file)
    pad = max(len(k) for k in info)
    for k, v in info.items():
        if isinstance(v, float):
            print(f"  {k:<{pad}}  {v:.3f}")
        else:
            print(f"  {k:<{pad}}  {v}")


def _run_freq(args):
    from .freq import frequency
    res = frequency(args.cugen_file)
    if args.out:
        import pandas as pd
        df = pd.DataFrame({
            "variant_idx": range(res["n_variants"]),
            "mu_x": res["mu_x"],
            "sxx": res["sxx"],
            "maf": res["maf"],
        })
        df.to_csv(args.out, sep="\t", index=False)
        print(f"wrote {args.out} ({len(df):,} rows)", file=sys.stderr)
    else:
        print(f"n_samples={res['n_samples']:,}  n_variants={res['n_variants']:,}")
        print(f"  maf  mean={res['maf'].mean():.4f}  "
              f"min={res['maf'].min():.4f}  max={res['maf'].max():.4f}")
        print(f"  sxx  mean={res['sxx'].mean():.2f}  "
              f"min={res['sxx'].min():.2f}  max={res['sxx'].max():.2f}")


def _run_workflow(args):
    from .config import (load_and_validate_config, write_example_config)
    from .workflow import ultra_workflow
    if args.example:
        write_example_config(args.example)
        print(f"wrote example config → {args.example}", file=sys.stderr)
        return
    if args.validate:
        cfg = load_and_validate_config(args.validate)
        print(json.dumps({"ok": True, "name": cfg["general"]["name"]}))
        return
    if args.dry_run:
        res = ultra_workflow(args.dry_run, dry_run=True)
        print(json.dumps({
            "ok": True,
            "phenotypes": [p[0] for p in res["phenotypes"]],
            "elapsed_s": res["elapsed_s"],
        }, indent=2))
        return
    print("workflow: provide --example, --validate, or --dry-run "
          "(or pass a JSON path directly: `cugen config.json`)",
          file=sys.stderr)
    sys.exit(2)


def _run_gwas(args):
    cfg = {
        "general": {"name": Path(args.out).stem, "output_dir": args.out},
        "gpu": {"n_workers_gwas": args.workers, "use_pinned_reader": True},
        "data": {
            "cugen_dir": args.cugen_dir,
            "imputed_cugen_dir": args.imputed_dir,
        },
        "phenotype": {"mode": "single", "file": args.pheno,
                      "type": args.pheno_type, "name": Path(args.pheno).stem},
        "covariates": {"columns": args.covariates} if args.covariates else {},
        "step5_finemapping": {"run": not args.no_finemapping},
    }
    from .workflow import ultra_workflow
    ultra_workflow(cfg)


def _run_qc(args):
    from ._stubs import _stub
    _stub("cli.qc (v0.2 roadmap)")


def _run_score(args):
    from ._stubs import _stub
    _stub("cli.score (v0.2 roadmap)")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="cugen",
        description=(
            "cugen — GPU-accelerated UltraLasso / UltraSuSiE pipeline. "
            "Pass a JSON config to run the full workflow; or use a subcommand."
        ),
    )
    subparsers = parser.add_subparsers(dest="command")

    _add_prep(subparsers)
    _add_info(subparsers)
    _add_freq(subparsers)
    _add_workflow(subparsers)
    _add_gwas(subparsers)
    _add_qc(subparsers)
    _add_score(subparsers)

    argv = argv if argv is not None else sys.argv[1:]

    # Shortcut: a single JSON path → run workflow
    if len(argv) == 1 and argv[0].endswith(".json") and Path(argv[0]).exists():
        from .workflow import ultra_workflow
        ultra_workflow(argv[0])
        return

    args = parser.parse_args(argv)
    cmd = args.command
    if cmd is None:
        parser.print_help()
        return
    dispatch = {
        "prep": _run_prep,
        "info": _run_info,
        "freq": _run_freq,
        "workflow": _run_workflow,
        "gwas": _run_gwas,
        "qc": _run_qc,
        "score": _run_score,
    }
    dispatch[cmd](args)


if __name__ == "__main__":
    main()
