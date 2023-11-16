import numpy as np

from tempor.benchmarks import utils


def test_generate_score():
    out = utils.generate_score(metric=np.asarray([1.0, 2.0, 3.0]))
    assert len(out) == 2

    mean, ci = out
    np.testing.assert_equal(mean, 2.0)
    np.testing.assert_almost_equal(ci, 0.9239, 3)


def test_print_score():
    out = utils.print_score(score=(1, 2))
    assert isinstance(out, str)
    assert out == "1 +/- 2"
