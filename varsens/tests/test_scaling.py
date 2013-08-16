from varsens    import scale
from nose.tools import *
from numpy      import *

def test_linear_scaling():
    n      = 5
    points = array(range(n))/(n-1.0)
    scaled = scale(points, array([[-100]*n, [100]*n]))
    answer = [-100.0, -50.0, 0.0, 50.0, 100.0]
    for i in range(n):
        assert_almost_equal(answer[i], scaled[i])