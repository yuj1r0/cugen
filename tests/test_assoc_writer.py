"""Step-4 sumstats writer and binary case counts (CPU only).

A POS column upcast to float64 by the gidx left-join (any annotation row
missing -> NaN) used to be written with float_format='%.6g', i.e. rounded to 6
significant digits (16057417 -> 1.60574e+07). CHR/POS/gidx are now exact Int64.
"""
import numpy as np
import pandas as pd

from cugen.assoc import _exact_int_columns, _mac_case_minor, _write_sumstats


def _toy():
    # POS as float64 with a NaN, exactly what the left-join on gidx produces.
    return pd.DataFrame({
        "CHR": ["22", "22", "22", "6"],
        "POS": [16057417.0, np.nan, 51234567.0, 32470486.0],
        "ID": ["rs1", None, "rs3", "DRB3_101"],
        "BETA": [0.0181217123, -1.23456789e-5, 2.5, -0.5],
        "P": [0.482883, 1e-300, 3.14159265e-12, 1.0],
        "MAC": [67138.8, 1.0, 20.0, 5.5],
        "gidx": [6727693, 6727692, 6727700, 1],
    })


def test_pos_written_as_exact_integer(tmp_path):
    out = tmp_path / "chr22_sumstats.tsv"
    _write_sumstats(_toy(), str(out))
    back = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert back["POS"].tolist() == ["16057417", "", "51234567", "32470486"]
    assert back["gidx"].tolist() == ["6727693", "6727692", "6727700", "1"]
    rt = pd.read_csv(out, sep="\t")
    assert rt["POS"].iloc[[0, 2, 3]].astype("int64").tolist() == [16057417, 51234567, 32470486]
    assert np.isnan(rt["POS"].iloc[1])


def test_float_chr_written_as_integer(tmp_path):
    df = pd.DataFrame({"CHR": [22.0, np.nan], "POS": [1.0, 2.0], "gidx": [5.0, 6.0]})
    out = tmp_path / "s.tsv"
    _write_sumstats(df, str(out))
    back = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    assert back["CHR"].tolist() == ["22", ""]
    assert back["gidx"].tolist() == ["5", "6"]


def test_float_statistics_keep_production_format(tmp_path):
    out = tmp_path / "s.tsv"
    _write_sumstats(_toy(), str(out))
    back = pd.read_csv(out, sep="\t", dtype=str, keep_default_na=False)
    for col in ("BETA", "P", "MAC"):
        assert back[col].tolist() == ["%.6g" % v for v in _toy()[col]]


def test_int_pos_unchanged_and_noninteger_never_rounded():
    df = pd.DataFrame({"POS": np.array([1, 2, 123456789], dtype=np.int64)})
    assert _exact_int_columns(df.copy())["POS"].tolist() == [1, 2, 123456789]
    df = pd.DataFrame({"POS": [123456789.5, np.nan]})
    assert _exact_int_columns(df)["POS"].tolist() == ["123456789.5", ""]


def test_old_format_was_lossy():
    # documents the bug: the old writer turned 16057417 into 16057400
    assert float("%.6g" % 16057417.0) == 16057400.0


def test_mac_case_minor():
    n_case = 100
    mac_case = np.array([5.0, 195.0, 100.0, 0.0, 200.0])
    assert _mac_case_minor(mac_case, n_case).tolist() == [5.0, 5.0, 100.0, 0.0, 0.0]
