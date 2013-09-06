from varsens    import *
from nose.tools import *
import numpy

def test_dimensionality():
    k = 11
    n = 13
    x = Sample(k, n, lambda x: x, False)

    assert_equal(x.k, k)
    assert_equal(x.n, n)
    assert_equal(x.M_1.shape[0], n)
    assert_equal(x.M_1.shape[1], k)
    assert_equal(x.M_2.shape[0], n)
    assert_equal(x.M_2.shape[1], k)
    assert_equal(x.N_j.shape[0], k)
    assert_equal(x.N_j.shape[1], n)
    assert_equal(x.N_j.shape[2], k)
    assert_equal(x.N_nj.shape[0], k)
    assert_equal(x.N_nj.shape[1], n)
    assert_equal(x.N_nj.shape[2], k)

def test_range():
    k = 7
    n = 11
    x = Sample(k, n, lambda x: x, False)

    for y in x.M_1.flat:
        assert(y >= 0.0)
        assert(y <= 1.0)

    for y in x.M_2.flat:
        assert(y >= 0.0)
        assert(y <= 10)

    for y in x.N_j.flat:
        assert(y >= 0.0)
        assert(y <= 10)

    for y in x.N_nj.flat:
        assert(y >= 0.0)
        assert(y <= 10)

def test_resampling():
    k=3
    n=5
    x = Sample(k, n, lambda x: x, False)
    for i in range(k):
        assert_almost_equal(numpy.sum(x.M_1.T[i]-x.N_j[i].T[i]), 0.0)
        assert_almost_equal(numpy.sum(x.M_2.T[i]-x.N_nj[i].T[i]), 0.0)
        for j in range(k):
            if j != i:
                assert_almost_equal(numpy.sum(x.M_1.T[j]-x.N_nj[i].T[j]), 0.0)
                assert_almost_equal(numpy.sum(x.M_2.T[j]-x.N_j[i].T[j]), 0.0)

def test_expected():
    k = 17
    n = 1024
    x = Sample(k, n, lambda x: x, False)

    assert_almost_equal(numpy.sum(x.M_1) / x.n / x.k,       0.5, places=2)
    assert_almost_equal(numpy.sum(x.M_2) / x.n / x.k,       0.5, places=2)
    assert_almost_equal(numpy.sum(x.N_j) / x.n / x.k / x.k, 0.5, places=2)
    assert_almost_equal(numpy.sum(x.N_nj)/ x.n / x.k / x.k, 0.5, places=2)

def test_flat():
    k = 5
    n = 13

    s = Sample(k, n, lambda x: x, False)

    assert_equal(s.flat().shape[0], n*(2*k+2))
    assert_equal(s.flat().shape[1], k)
    assert_equal(numpy.sum(s.M_1[0]),        numpy.sum(s.flat()[0] ))
    assert_equal(numpy.sum(s.N_nj[k-1][-1]), numpy.sum(s.flat()[-1]))
