"""cugen.background on CPU against dense numpy references.

Small simulated data, a few seconds. Each engine's result is compared with a direct,
independent dense implementation (solves instead of eigendecompositions, explicit
leave-one-out refits, a dense GLS).
"""
import numpy as np
import pytest

import cugen as cg
from conftest import a2_counts
from cugen import background as bg
from cugen.background._kernels import KernelSet


@pytest.fixture(autouse=True)
def _cpu_only(monkeypatch):
    """These are the CPU tests: device="auto" must resolve to the CPU even on a GPU node."""
    monkeypatch.setenv("CUGEN_DEVICE", "cpu")


# ----------------------------------------------------------------------------- references
def ref_std(G, Q, ncov):
    """REGENIE genotype standardisation, dense."""
    X = G.astype(float)
    miss = G == 3
    X[miss] = np.nan
    mu = np.nanmean(X, 1)
    X = np.where(miss, mu[:, None], X)
    X = X - (X @ Q) @ Q.T
    sc = np.sqrt((X ** 2).sum(1) / (X.shape[1] - ncov))
    return X / sc[:, None]


def basis(C, n):
    Q, _ = np.linalg.qr(np.column_stack([np.ones(n), C]))
    return Q


# ----------------------------------------------------------------------------- CuGenie
def ref_cugenie(G, chrom, y, C, bsize, K, h2):
    n = G.shape[1]
    Q = basis(C, n)
    ncov = Q.shape[1]
    yr = y - Q @ (Q.T @ y)
    ys = yr / np.sqrt((yr ** 2).sum() / (n - ncov))
    X = ref_std(G, Q, ncov)
    M = G.shape[0]
    blocks = []
    for c in np.unique(chrom):
        ii = np.flatnonzero(chrom == c)
        for s in range(ii[0], ii[-1] + 1, bsize):
            blocks.append((c, s, min(s + bsize, ii[-1] + 1)))
    t = n // K
    b = np.cumsum([0] + [t] * (K - 1) + [n - t * (K - 1)])
    folds = [np.arange(b[i], b[i + 1]) for i in range(K)]
    lam = M * (1 - h2) / h2
    R = len(h2)
    W = np.zeros((n, len(blocks) * R))
    for j, (c, s, e) in enumerate(blocks):
        Xb = X[s:e]
        for f in folds:
            tr = np.setdiff1d(np.arange(n), f)
            for r in range(R):
                A = Xb[:, tr] @ Xb[:, tr].T + lam[r] * np.eye(e - s)
                beta = np.linalg.solve(A, Xb[:, tr] @ ys[tr])
                W[f, j * R + r] = Xb[:, f].T @ beta
    W = (W - W.mean(0)) / W.std(0, ddof=1)
    tau = len(blocks) * R * (1 - h2) / h2
    preds = np.zeros((n, R))
    betas = []
    for f in folds:
        tr = np.setdiff1d(np.arange(n), f)
        bf = [np.linalg.solve(W[tr].T @ W[tr] + t_ * np.eye(W.shape[1]), W[tr].T @ ys[tr]) for t_ in tau]
        betas.append(bf)
        for r in range(R):
            preds[f, r] = W[f] @ bf[r]
    j = int(np.argmin(((preds - ys[:, None]) ** 2).mean(0)))
    colchr = np.repeat([b_[0] for b_ in blocks], R)
    nC = int(chrom.max())
    L = np.zeros((n, nC))
    for i, f in enumerate(folds):
        bb = betas[i][j]
        tot = W[f] @ bb
        for c in range(1, nC + 1):
            L[f, c - 1] = tot - W[f][:, colchr == c] @ bb[colchr == c]
    return L, j


def test_cugenie_masked_equals_dense_reference(cohort):
    h2 = np.array([0.1, 0.5, 0.9])
    L = cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=25, folds=4, h2_l0=h2, h2_l1=h2,
                       verbose=False)
    Lref, j = ref_cugenie(cohort["G"], cohort["chrom"], cohort["y"], cohort["C"], 25, 4, h2)
    assert L.meta["level1"]["y"]["tau_index"] == j
    np.testing.assert_allclose(L.offsets[:, :, 0], Lref, rtol=0, atol=1e-9)
    Q = basis(cohort["C"], len(cohort["y"]))
    yhat = Q @ (Q.T @ cohort["y"])
    np.testing.assert_allclose(L.predictions[:, 0, 0], yhat + L.scale_y[0] * Lref[:, 0], atol=1e-9)
    assert L.chromosomes.tolist() == [1, 2, 3] and L.meta["chromosomes"] == [1, 2, 3]
    assert "A1" in L.meta["counted_allele"]


def test_cugenie_refit_and_random_folds_run(cohort):
    out = cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=30, loco="both",
                         fold_kind="random", seed=1, verbose=False)
    assert set(out) == {"masked", "refit"}
    assert np.isfinite(out["refit"].predictions).all()


def test_cugenie_multi_phenotype_equals_single(cohort):
    Y = np.column_stack([cohort["y"], cohort["y"][::-1].copy()])
    both = cg.fit_cugenie(cohort["bed"], Y, cohort["C"], block_size=40, verbose=False)
    one = cg.fit_cugenie(cohort["bed"], Y[:, 1], cohort["C"], block_size=40, verbose=False)
    np.testing.assert_allclose(both.predictions[:, :, 1], one.predictions[:, :, 0], atol=1e-10)


def test_cugenie_chromosomes_subset(cohort):
    full = cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=30, verbose=False)
    sub = cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=30, chromosomes=[1, 3],
                         verbose=False)
    assert sub.chromosomes.tolist() == [1, 3] and sub.n_chrom == 2
    assert sub.meta["M"] == int(np.isin(cohort["chrom"], [1, 3]).sum())
    assert sub.column(3) == 1
    with pytest.raises(KeyError):
        sub.column(2)
    # same as the full fit with chromosomes=range(1, max+1) explicitly
    same = cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=30, chromosomes=range(1, 4),
                          verbose=False)
    np.testing.assert_array_equal(same.predictions, full.predictions)


# ----------------------------------------------------------------------------- LocoPredictions
def test_loco_roundtrip_keeps_chromosomes_and_long_ids(tmp_path):
    n = 5
    ids = [f"sample_with_a_long_identifier_{i:04d}" for i in range(n)]       # > 16 characters
    lp = bg.LocoPredictions(np.arange(n * 2.0).reshape(n, 2), np.arange(n), ids, chromosomes=[2, 7])
    p = lp.save(str(tmp_path / "x"))[0]
    back = bg.LocoPredictions.load(p)
    assert back.fid_order.tolist() == ids
    assert back.chromosomes.tolist() == [2, 7]
    np.testing.assert_array_equal(back.residual(7), lp.residual(7))
    # files written before the field existed read as chromosomes 1..C
    np.savez(tmp_path / "old.npz", predictions=np.zeros((n, 3), np.float32), y_original=np.zeros(n, np.float32))
    assert bg.LocoPredictions.load(str(tmp_path / "old.npz")).chromosomes.tolist() == [1, 2, 3]
    with pytest.raises(ValueError):
        bg.LocoPredictions(np.zeros((n, 2)), np.zeros(n), chromosomes=[3, 1])


# ----------------------------------------------------------------------------- CuGenie-K
def test_cugenie_k_loocv_equals_explicit_leave_one_out(cohort):
    lam = np.logspace(-2, 2, 9)
    G, chrom, y, C = cohort["G"], cohort["chrom"], cohort["y"], cohort["C"]
    n = len(y)
    res = cg.fit_cugenie_k(cohort["bed"], y, C, lam_grid=lam, offset="both", verbose=False)
    Q = basis(C, n)
    ncov = Q.shape[1]
    yr = y - Q @ (Q.T @ y)
    ys = yr / np.sqrt((yr ** 2).sum() / (n - ncov))
    X = ref_std(G, Q, ncov)
    c = 2
    Zm = X[chrom != c]
    Mm = Zm.shape[0]
    j = int(np.argmin(res["loo"].meta["loocv_curves"][c - 1][:, 0]))
    lj = lam[j]
    Kc = Zm.T @ Zm / Mm
    H = Q @ Q.T + Kc @ np.linalg.inv(Kc + lj * np.eye(n))
    np.testing.assert_allclose(res["insample"].offsets[:, c - 1, 0], H @ ys - (Q @ Q.T) @ ys, atol=1e-8)
    Xd = np.column_stack([Q, Zm.T / np.sqrt(Mm)])
    Pen = np.diag(np.r_[np.zeros(ncov), np.full(Mm, lj)])
    rl = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        th = np.linalg.solve(Xd[keep].T @ Xd[keep] + Pen, Xd[keep].T @ ys[keep])
        rl[i] = ys[i] - Xd[i] @ th
        if i in (0, 17, 101, n - 1):
            assert abs((Xd[i] @ th) - res["loo"].offsets[i, c - 1, 0]) < 1e-8
    np.testing.assert_allclose(res["loo"].meta["loocv_curves"][c - 1][j, 0], (rl ** 2).mean(), rtol=1e-9)


def test_cugenie_k_null_tol(cohort):
    a = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], verbose=False)
    b = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], null_tol=1e-10, verbose=False)
    lam = np.asarray(a.meta["lam_grid"])
    ok = lam >= 1e-3
    ca, cb = np.asarray(a.meta["loocv_curves"]), np.asarray(b.meta["loocv_curves"])
    np.testing.assert_allclose(ca[:, ok], cb[:, ok], rtol=1e-8)
    assert sum(a.meta["M_per_chr"].values()) == a.meta["M"]
    assert a.meta["lam_stable_floor"] == 1e-5 and a.meta["n_lam_star_below_stable_floor"] >= 0


def test_cugenie_k_named_lambda_grids(cohort):
    w = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], verbose=False)
    st = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], lam_grid="stable", verbose=False)
    assert len(w.meta["lam_grid"]) == 101 and min(w.meta["lam_grid"]) == pytest.approx(1e-7)
    assert len(st.meta["lam_grid"]) == 81 and min(st.meta["lam_grid"]) == pytest.approx(1e-5)
    explicit = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], lam_grid="wide", verbose=False)
    np.testing.assert_array_equal(explicit.offsets, w.offsets)
    with pytest.raises(ValueError):
        cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], lam_grid="nope", verbose=False)


def test_cugenie_k_low_memory_and_float32_match(cohort):
    a = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], low_memory=False, verbose=False)
    b = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], low_memory=True, verbose=False)
    np.testing.assert_allclose(a.offsets, b.offsets, atol=1e-10)
    f = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], gram_dtype="float32", verbose=False)
    assert np.abs(f.offsets - a.offsets).max() < 1e-4


# ----------------------------------------------------------------------------- UltraLOCO
def test_ultraloco_equals_dense_gls(cohort):
    G, chrom, y, C = cohort["G"], cohort["chrom"], cohort["y"], cohort["C"]
    n = len(y)
    m = cg.fit_ultraloco(cohort["cg"], y, C, verbose=False)
    ss = m.gwas(test_dtype="float64", maf_min=None)
    A2 = a2_counts(G)
    Zs, Ms = [], []
    for v in range(A2.shape[0]):
        g = A2[v].astype(float)
        miss = A2[v] == 3
        p = g[~miss].mean() / 2
        Zs.append(np.where(miss, 0, (g - 2 * p) / np.sqrt(2 * p * (1 - p))))
        Ms.append((~miss).astype(float))
    Zs, Ms = np.array(Zs), np.array(Ms)
    X0 = np.column_stack([np.ones(n), C])
    for c in (1, 3):
        o = chrom != c
        K = (Zs[o].T @ Zs[o]) / (Ms[o].T @ Ms[o])
        # GLS on the PSD part of K_-c (a pairwise-count GRM on ~100 markers is not PSD)
        s_, U_ = np.linalg.eigh((K + K.T) / 2)
        K = (U_ * np.maximum(s_, 0)) @ U_.T
        info = m.chrom_info[c]
        Vi = np.linalg.inv(K + info["delta"] * np.eye(n))
        for v in np.flatnonzero(chrom == c):
            g = A2[v].astype(float)
            miss = A2[v] == 3
            x = np.where(miss, 0, g - g[~miss].mean())
            X = np.column_stack([X0, x])
            A = X.T @ Vi @ X
            beta = np.linalg.solve(A, X.T @ Vi @ y)[-1]
            se = np.sqrt(info["sg2"] * np.linalg.inv(A)[-1, -1])
            row = ss[ss.gidx == v].iloc[0]
            assert abs(row.Z - beta / se) < 2e-6 * max(1, abs(beta / se))       # Z stored as float32
            assert abs(row.BETA - beta) < 2e-6 * max(1e-2, abs(beta))
    assert list(ss.columns) == ["CHR", "POS", "BETA", "SE", "Z", "P", "MAF", "gidx"]   # no annotation given
    assert ss.attrs["counted_allele"].startswith("cugen ALT")


def test_ultraloco_reml_is_a_maximum(cohort):
    m = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], verbose=False)
    m.gwas(chromosomes=[2])
    K = m.kernels.loco(2)
    s, U = np.linalg.eigh(K)
    ys, Xs = U.T @ cohort["y"], U.T @ m.X0

    def ll(d):
        w = 1 / (np.maximum(s, 0) + d)
        A = (Xs * w[:, None]).T @ Xs
        b = (Xs * w[:, None]).T @ ys
        yPy = (w * ys * ys).sum() - b @ np.linalg.solve(A, b)
        nk = len(ys) - Xs.shape[1]
        return -0.5 * (nk * np.log(yPy / nk) + np.log(np.maximum(s, 0) + d).sum() + np.linalg.slogdet(A)[1])
    d0 = m.chrom_info[2]["delta"]
    if 1e-3 < d0 < 1e4:
        assert ll(d0) >= max(ll(d0 * 1.05), ll(d0 / 1.05)) - 1e-9


def test_ultraloco_reml_bounds_flag(cohort):
    m = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], verbose=False)
    ss = m.gwas(reml_bounds=(10.0, 1e4))
    assert all(v["at_bound"] and abs(v["delta"] - 10.0) < 1e-3 for v in ss.attrs["chrom_info"].values())
    assert ss.attrs["meta"]["n_chrom_reml_at_bound"] == 3


def test_ultraloco_gwas_is_repeatable_and_end_to_end(cohort):
    m = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], verbose=False)
    a = m.gwas()
    b = cg.ultraloco_gwas(m)
    np.testing.assert_array_equal(a.Z.to_numpy(), b.Z.to_numpy())
    e2e = cg.ultraloco_gwas(cohort["cg"], cohort["y"], cohort["C"], verbose=False)
    np.testing.assert_array_equal(e2e.Z.to_numpy(), a.Z.to_numpy())
    with pytest.raises(TypeError):
        cg.ultraloco_gwas(m, cohort["y"])


def test_ultraloco_chromosomes(cohort):
    m = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], chromosomes=[1, 2], verbose=False)
    ss = m.gwas(maf_min=None)
    assert sorted(ss.CHR.unique()) == [1, 2] and m.meta["M_grm"] <= int(np.isin(cohort["chrom"], [1, 2]).sum())


def test_kernelset_drop_recomputes(cohort):
    rows = cohort["cg"].select()
    ks = KernelSet(cohort["cg"], rows, cg.device("cpu"), "gcta", pairwise_count=True).build()
    k1 = ks.loco(2)
    ks.drop(2)
    np.testing.assert_allclose(ks.loco(2), k1, atol=1e-12)


def test_ultraloco_low_memory_and_modes(cohort):
    a = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], low_memory=False, verbose=False).gwas()
    b = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], low_memory=True, verbose=False).gwas()
    np.testing.assert_allclose(a.Z, b.Z, atol=1e-5)
    c = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], pairwise_count=False, verbose=False).gwas()
    r = cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], scaling="regenie", verbose=False).gwas()
    assert np.isfinite(c.Z).all() and np.isfinite(r.Z).all()


# ----------------------------------------------------------------------------- guards
def test_binary_phenotype_warns(cohort):
    yb = (cohort["y"] > np.median(cohort["y"])).astype(float)
    with pytest.warns(UserWarning, match="linear-probability"):
        cg.fit_cugenie(cohort["bed"], yb, cohort["C"], block_size=40, verbose=False)
    with pytest.warns(UserWarning, match="linear-probability"):
        cg.fit_cugenie_k(cohort["bed"], yb, cohort["C"], verbose=False)


def test_missing_phenotype_is_a_clear_error(cohort):
    y = cohort["y"].copy()
    y[3] = np.nan
    with pytest.raises(ValueError, match="missing"):
        cg.fit_cugenie(cohort["bed"], y, cohort["C"], verbose=False)


def test_estimates_and_resource_error(cohort):
    est = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], estimate_only=True)
    assert est["peak_bytes"] > 0 and "suggestions" in est and est["fits"]
    assert est["peak_bytes"] >= est["work_bytes"]
    for k in ("method", "device", "peak_bytes", "work_bytes", "budget_bytes", "fits", "flops",
              "flops_by_stage", "dominant", "suggestions"):
        assert k in est
    calls = (lambda: cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], memory_budget="100KB",
                                      low_memory=False, verbose=False),
             lambda: cg.fit_ultraloco(cohort["cg"], cohort["y"], cohort["C"], memory_budget="100KB",
                                      verbose=False),
             lambda: cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], memory_budget="100KB",
                                    verbose=False),
             lambda: bg.gwas_blocked(cohort["bed"], cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"],
                                                                   verbose=False),
                                     memory_budget="100KB", verbose=False))
    for fn in calls:
        with pytest.raises(cg.CugenResourceError) as e:
            fn()
        assert "Options that reduce memory" in str(e.value) and e.value.suggestions


def test_low_memory_auto_picks_low_memory_when_tight(cohort):
    big = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], estimate_only=True)
    tight = int(big["work_bytes"] * 0.995)
    est = cg.fit_cugenie_k(cohort["bed"], cohort["y"], cohort["C"], estimate_only=True, memory_budget=tight)
    assert est["low_memory"] is True


def test_logging_not_print(cohort, capsys, caplog):
    import logging
    with caplog.at_level(logging.INFO, logger="cugen.background"):
        cg.fit_cugenie(cohort["bed"], cohort["y"], cohort["C"], block_size=40, verbose=True)
    assert capsys.readouterr().out == ""
    assert any("level 1" in r.getMessage() for r in caplog.records)
