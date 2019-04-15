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


def test_log_transform():
    return
    samp_var = [[i + 1 for i in range(10)] for j in range(10)]
    print(samp_var)
    tmpdir = gettempdir()
    var_number = 3
    transformed = stats.log_transform(samp_var, tmpdir, var_number)
    print(transformed)
