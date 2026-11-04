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
