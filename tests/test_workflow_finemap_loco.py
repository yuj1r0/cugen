"""ultra_workflow: the LOCO handed to single-worker fine-mapping after step 3."""
from pathlib import Path

import numpy as np

from cugen.workflow import _loco_for_finemapping


def test_bare_array_uses_saved_npz(tmp_path):
    npz = tmp_path / "loco_predictions.npz"
    np.savez(npz, predictions=np.zeros((3, 22)), y_original=np.zeros(3))
    out = _loco_for_finemapping(np.zeros((3, 22)), npz, {"y_original": np.zeros(3)})
    assert out == str(npz)


def test_bare_array_without_npz_becomes_dict(tmp_path):
    pred = np.arange(6.0).reshape(3, 2)
    y = np.array([1.0, 2.0, 3.0])
    out = _loco_for_finemapping(pred, tmp_path / "missing.npz", {"y_original": y})
    assert isinstance(out, dict) and set(out) == {"predictions", "y_original"}
    assert out["predictions"] is pred and out["y_original"] is y


def test_paths_and_dicts_unchanged(tmp_path):
    d = {"predictions": np.zeros((2, 2)), "y_original": np.zeros(2)}
    assert _loco_for_finemapping(d, tmp_path / "x.npz") is d
    assert _loco_for_finemapping("a/b.npz", tmp_path / "x.npz") == "a/b.npz"
    p = Path("a/b.npz")
    assert _loco_for_finemapping(p, tmp_path / "x.npz") is p
    assert _loco_for_finemapping(None, tmp_path / "x.npz") is None
