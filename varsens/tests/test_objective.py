from varsens    import *
from nose.tools import *
import numpy

def invert(x): return 1.0-x

def __init__(self, k, n, sample, objective_func, verbose=True):
    self.k              = k
    self.n              = n
    self.sample         = sample
    self.objective_func = objective_func
    self.verbose        = verbose
    self.fM_1           = None
    self.fM_2           = None
    self.fN_j           = None
    self.fN_nj          = None

def test_invert():
    k = 8
    n = 23
    s = Sample(k, n, lambda x: x, False)
    o = Objective(k, n, s, invert, False)
    assert_almost_equal(numpy.sum(1.0 - o.fM_1 - s.M_1), 0.0)
    assert_almost_equal(numpy.sum(1.0 - o.fM_2 - s.M_2), 0.0)
    assert_almost_equal(numpy.sum(1.0 - o.fN_j - s.N_j), 0.0)
    assert_almost_equal(numpy.sum(1.0 - o.fN_nj - s.N_nj), 0.0)

def test_shape():
    k = 8
    n = 23
    s = Sample(k, n, lambda x: x, False)
    o = Objective(k, n, s, invert, False)
    assert_equal(o.fM_1.shape[0], n)
    assert_equal(o.fM_1.shape[1], k)
    assert_equal(o.fM_2.shape[0], n)
    assert_equal(o.fM_2.shape[1], k)
    assert_equal(o.fN_j.shape[0], k)
    assert_equal(o.fN_j.shape[1], n)
    assert_equal(o.fN_j.shape[2], k)
    assert_equal(o.fN_nj.shape[0], k)
    assert_equal(o.fN_nj.shape[1], n)
    assert_equal(o.fN_nj.shape[2], k)

def test_empty():
    k = 8
    n = 23
    s = Sample(k, n, lambda x: x, False)
    o = Objective(k, n, s, None, False)
    assert_equal(o.fM_1, None)
    assert_equal(o.fM_2, None)
    assert_equal(o.fN_j, None)
    assert_equal(o.fN_nj, None)