import numpy as np

from hrag.qaf import ALPHA_GRID, FEATURE_NAMES, QAFModel, oracle_alpha, query_features


def _f(q, dense=None, sparse=None, vocab=None):
    v = query_features(q, dense or [], sparse or [], idf=lambda t: 1.0, tokenizer_vocab=vocab or set())
    return dict(zip(FEATURE_NAMES, v))


def test_feature_names_and_grid():
    assert len(FEATURE_NAMES) == 12
    assert ALPHA_GRID[0] == 0.0 and ALPHA_GRID[-1] == 1.0 and len(ALPHA_GRID) == 11


def test_identifier_query():
    v = query_features("RFC 9110 section 15.5.1", [], [], idf=lambda t: 1.0, tokenizer_vocab=set())
    assert v.shape == (12,)
    f = dict(zip(FEATURE_NAMES, v))
    assert f["id_ratio"] > 0.4
    assert f["sparse_top"] == 0
    assert f["len_tokens"] == 4


def test_natural_query():
    f = _f("how do cats communicate", vocab={"how", "do", "cats", "communicate"})
    assert f["id_ratio"] == 0 and f["oov_ratio"] == 0 and f["caps_ratio"] == 0


def test_empty_query_all_zero():
    assert all(x == 0 for x in _f("").values())


def test_ranking_features():
    d = [("a", 0.9), ("b", 0.5)]
    s = [("a", 3.0), ("c", 1.0)]
    f = _f("x", d, s)
    assert f["dense_top"] == 0.9 and abs(f["dense_margin"] - 0.4) < 1e-9
    assert f["sparse_top"] == 3.0 and f["sparse_margin"] == 2.0
    assert abs(f["overlap_at10"] - 0.1) < 1e-9


def test_oracle_alpha_prefers_correct_side():
    # relevant doc only in sparse -> alpha < 0.5; only in dense -> alpha > 0.5.
    # RRF is rank-based, so every alpha on that side ties and the tie-break picks the
    # grid point closest to 0.5 (0.4 / 0.6), then the smaller alpha.
    dense = [("x", 1.0), ("d1", 0.5)]
    sparse = [("y", 1.0), ("s1", 0.5)]
    assert oracle_alpha(dense, sparse, {"s1": 1}) == 0.4
    assert oracle_alpha(dense, sparse, {"d1": 1}) == 0.6
    assert oracle_alpha(dense, sparse, {}) == 0.5
    assert oracle_alpha(dense, sparse, {"zzz": 1}) == 0.5


def test_oracle_alpha_tie_toward_half():
    d = [("a", 1.0)]
    s = [("a", 1.0)]
    assert oracle_alpha(d, s, {"a": 1}) == 0.5


def test_model_fit_predict_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.uniform(size=(200, 12))
    y = np.clip(0.8 - 0.6 * X[:, 0], 0, 1)
    m = QAFModel().fit(X, y)
    p = m.predict(X)
    assert p.min() >= 0 and p.max() <= 1
    assert np.abs(p - y).max() < 0.1
    assert m.predict(X[0]).shape == (1,)
    ct = m.coef_table()
    assert [n for n, _ in ct] == FEATURE_NAMES
    m.save(tmp_path / "m.joblib")
    m2 = QAFModel.load(tmp_path / "m.joblib")
    assert np.allclose(m2.predict(X), p)


def test_oracle_alpha_ties_pick_closest_to_half():
    # relevant at rank 1 of sparse only: every alpha < 0.5 ties; closest to 0.5 is 0.4
    assert oracle_alpha([("d", 1.0)], [("s", 1.0)], {"s": 1}) == 0.4
