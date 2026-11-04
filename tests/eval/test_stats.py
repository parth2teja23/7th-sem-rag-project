from hrag.eval.stats import paired_bootstrap, paired_ttest


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
