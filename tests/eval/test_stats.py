import pytest

from hrag.eval.stats import paired_bootstrap, paired_bootstrap_ci, paired_ttest


def test_bootstrap_clear_win():
    assert paired_bootstrap([0.0] * 50, [1.0] * 50) < 0.01


def test_bootstrap_identical():
    x = [0.3, 0.5, 0.7] * 10
    assert paired_bootstrap(x, list(x)) > 0.4


def test_ttest():
    a = [0.1 * i for i in range(20)]
    b = [v + 0.5 + 0.01 * (i % 3) for i, v in enumerate(a)]
    assert paired_ttest(a, b) < 0.001


def test_bootstrap_direction():
    assert paired_bootstrap([1.0] * 50, [0.0] * 50) > 0.99


def test_ttest_direction_and_degenerate():
    assert paired_ttest([0.5, 0.6, 0.7, 0.9], [0.1, 0.2, 0.3, 0.2]) > 0.99
    assert paired_ttest([0.0] * 5, [0.0] * 5) == 1.0
    assert paired_ttest([0.0] * 5, [0.3] * 5) == 0.0
    assert paired_ttest([0.3] * 5, [0.0] * 5) == 1.0


def test_bootstrap_ci_constant_shift_is_degenerate():
    a = [0.1 * (i % 7) for i in range(40)]
    b = [v + 0.25 for v in a]
    mean, lo, hi = paired_bootstrap_ci(a, b)
    assert mean == pytest.approx(0.25) and lo == pytest.approx(0.25) and hi == pytest.approx(0.25)


def test_bootstrap_ci_identical_is_zero():
    x = [0.3, 0.5, 0.7] * 10
    assert paired_bootstrap_ci(x, list(x)) == (0.0, 0.0, 0.0)


def test_bootstrap_ci_brackets_mean_and_is_seeded():
    a = [0.0] * 60
    b = [0.1 if i % 3 else -0.05 for i in range(60)]
    mean, lo, hi = paired_bootstrap_ci(a, b)
    assert lo < mean < hi
    assert (mean, lo, hi) == paired_bootstrap_ci(a, b)
