import cutie.utils as utils
import numpy as np

from tempfile import gettempdir
from pathlib import Path


def test_calculate_intersection():
    set_names = ['Set1', 'Set2', 'Set3']
    sets = [
        {(1, 2), (3, 4)},
        {(3, 4), (4, 5)},
        {(4, 5), (1, 2)}
    ]
    log = Path(gettempdir()) / 'intersection_log.txt'

    r_sets = utils.calculate_intersection(set_names, sets, log)

    assert r_sets["['Set1', 'Set2']"] == {(3, 4)}
    assert r_sets["['Set1', 'Set3']"] == {(1, 2)}
    assert r_sets["['Set2', 'Set3']"] == {(4, 5)}

    # Check it returns an empty list
    assert not utils.calculate_intersection([], [], log)


def test_get_param():
    n_samples = 4
    samp_var1 = [[j for j in range(15)] for i in range(n_samples)]
    samp_var2 = [[j for j in range(8)] for i in range(n_samples)]
    n_var1, n_var2, n_samp = utils.get_param(samp_var1, samp_var2)

    assert n_var1 == 15
    assert n_var2 == 8
    assert n_samp == n_samples

    zeros = np.zeros((3, 5, 2), dtype=np.complex128)

    n_var1, n_var2, n_samp = utils.get_param([[]], [[]])
    assert n_var1 == 0
    assert n_var2 == 0
    assert n_samp == 0


def test_remove_nans():
    # Check Empty
    var1 = []
    var2 = []
    nvar1, nvar2 = utils.remove_nans(var1, var2)
    assert not nvar1.any()
    assert not nvar2.any()

    # Check some nan
    var1 = [1, np.nan, np.nan, 4, 5, 6, np.nan]
    var2 = [np.nan, 2, np.nan, 4, 5, np.nan, 7]

    nvar1, nvar2 = utils.remove_nans(var1, var2)
    assert (nvar1 == np.array([4, 5])).all()
    assert (nvar2 == np.array([4, 5])).all()

    # Check all nan
    var1 = [np.nan, np.nan, np.nan, np.nan]
    var2 = [np.nan, np.nan, np.nan, 7]

    nvar1, nvar2 = utils.remove_nans(var1, var2)
    assert not nvar1.any()
    assert not nvar2.any()

    # Check no nan
    var1 = [i for i in range(10)]
    var2 = [i * i for i in range(10)]

    nvar1, nvar2 = utils.remove_nans(var1, var2)
    assert (nvar1 == np.array(var1)).all()
    assert (nvar2 == np.array(var2)).all()
