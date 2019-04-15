import cutie.statistics as stats
import numpy as np
from tempfile import gettempdir
from pytest import raises


def test_zero_replacement():
    # Test no zeros
    samp_var = [[i + 1 for i in range(10)] for j in range(10)]
    result = stats.zero_replacement(samp_var)
    assert result == 0.5

    # Test all zeros
    samp_var = [[0 for i in range(10)] for j in range(10)]
    # Should raise a value error
    with raises(ValueError):
        result = stats.zero_replacement(samp_var)

    # Test small numbers
    samp_var = [[float(i + 1) / 10 for i in range(10)] for j in range(10)]
    result = stats.zero_replacement(samp_var)
    assert result == (0.1 ** 2)


def test_multi_zeros():
    # Check no zeros are returned
    samp_var = np.array([np.array([i for i in range(10)]) for j in range(10)])
    samp_var_mr, samp_var_clr, samp_var_lclr, samp_var_varlog = stats.multi_zeros(samp_var)

    # Further checking of the values would just involve repeating code
    assert not samp_var_mr.any() == 0
    assert not samp_var_clr.any() == 0
    assert not samp_var_lclr.any() == 0

    # Test all zeros
    samp_var = [[0 for i in range(10)] for j in range(10)]
    # Should raise a value error
    with raises(ValueError):
        samp_var_mr, samp_var_clr, samp_var_lclr, samp_var_varlog = stats.multi_zeros(samp_var)


def test_log_transform():
    tmpdir = gettempdir() + '/'
    var_number = 3

    # Check no zeros are returned
    samp_var = np.array([np.array([i for i in range(10)]) for j in range(10)])
    transformed = stats.log_transform(samp_var, str(tmpdir), var_number)

    # Further checking of the values would just involve repeating code
    assert not transformed.any() == 0

    # Test all zeros
    samp_var = [[0 for i in range(10)] for j in range(10)]

    # Should raise a value error
    with raises(ValueError):
        transformed = stats.log_transform(samp_var, str(tmpdir), var_number)
