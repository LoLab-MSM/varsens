from nose.tools    import *
from numpy         import *
from varsens.scale import *

def test_linear_scaling():
    n      = 5
    points = array(range(n))/(n-1.0)
    scaled = linear_scale(points, array([-100]*n), array([100]*n))
    answer = [-100.0, -50.0, 0.0, 50.0, 100.0]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])

def test_linear_lower_bound():
    n      = 3
    scaled = linear_scale(array([0]*3), array([-100, -10, 1000]), array([100, 20, 2000]))
    answer = [-100.0, -10.0, 1000.0]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])

def test_linear_upper_bound():
    n      = 3
    scaled = linear_scale(array([1.0]*3), array([-100, -10, 1000]), array([100, 20, 2000]))
    answer = [100.0, 20.0, 2000.0]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])

def test_power_scaling():
    n      = 5
    points = array(range(n))/(n-1.0)
    scaled = power_scale(points, array([1]*n), array([100]*n))
    answer = [1.0, 10*sqrt(0.1), 10.0, 100*sqrt(0.1), 100.0 ]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])

def test_power_lower_bound():
    n      = 3
    scaled = power_scale(array([0]*n), array([1, 10, 1000]), array([100, 20, 2000]))
    answer = [1, 10, 1000]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])

def test_power_upper_bound():
    n      = 3
    scaled = power_scale(array([1.0]*n), array([1, 10, 1000]), array([100, 20, 2000]))
    answer = [100.0, 20.0, 2000.0]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])

def test_percentage_scaling():
    n      = 5
    points = array(range(n))/(n-1.0)
    scaled = percentage_scale(points, array([-10, -1, 0, 1, 20]), 33.0)
    answer = [-6.7,-0.835,0.0,1.165,26.6]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])

def test_magnitude_scaling():
    n      = 5
    points = array(range(n))/(n-1.0)
    scaled = magnitude_scale(points, array([1, 10, 100, 1000, 1e4]))
    answer = [0.001, sqrt(0.1), 100, sqrt(0.1)*1e5, 1e+07]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])