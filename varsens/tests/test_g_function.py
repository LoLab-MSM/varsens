from varsens    import *
from nose.tools import *
import numpy

########################################################
# Sobel's g-function for testing
# Eq (30)
def gi_function(xi, ai):
    return (numpy.abs(4.0*xi-2.0)+ai) / (1.0+ai)

# Eq (29)
def g_function(x, a):
    return numpy.prod([gi_function(xi, a[i]) for i,xi in enumerate(x)])

def g_scaling(x):
    return x # This is defined on the range [0..1]

model = [0, 0.5, 3, 9, 99, 99]

def g_objective(x): return g_function(x, model)

# Would be nice to have a legendre polynomial test as well
# http://www.jstor.org/stable/pdfplus/2676831.pdf
#from numpy.polynomial.legendre import legval

def test_g_function():
    # Analytical answer, Eq (34) divided by V(y), matches figure
    v = Varsens(g_objective, g_scaling, 6, 1024*10, verbose=False)
    estimate = v.sens * v.var_y
    truth    = 1.0/(3.0*((numpy.array(model) + 1.0)**2.0))

    for i in range(v.k):
        assert_almost_equal(truth[i], estimate[i], places=2)