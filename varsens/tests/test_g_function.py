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

def g_truth(model):
    return 1.0/(3.0*((numpy.array(model) + 1.0)**2.0))

def g_truth2(model, i, j):
    x = g_truth(model)
    return x[i]+x[j]+x[i]*x[j]

def g_truth_t(model, i):
    x = g_truth(model)


#def v_i2s(model, i2s):
#    def v_i(i):
#        return (v_truth(model))[i]
#    f = numpy.vectorize(v_i)
#    return numpy.prod(f(i2s))

# Would be nice to have a legendre polynomial test as well
# http://www.jstor.org/stable/pdfplus/2676831.pdf
#from numpy.polynomial.legendre import legval

def test_g_function():
    # Analytical answer, Eq (34) divided by V(y), matches figure
    v = Varsens(g_objective, g_scaling, 6, 1024*10, verbose=False)
    estimate  = v.sens   * v.var_y
    estimate2 = v.sens_2 * v.var_y
    truth    = g_truth(model)
    
    for i in range(v.k):
        assert_almost_equal(truth[i], estimate[i], places=2)
        for j in range(i+1, v.k):
            assert_almost_equal(g_truth2(model, i, j), estimate2[i,j], places=2)

def g_double_objective(x): return [g_function(x, model), g_function(x, model[::-1])]

def test_double_g_function():
    # Analytical answer, Eq (34) divided by V(y), matches figure
    v = Varsens(g_double_objective, g_scaling, 6, 1024*10, verbose=False)
    estimate = v.sens * v.var_y
    print estimate
    truth    = g_truth(model)

    for i in range(v.k):
        assert_almost_equal(truth[i], estimate[i][0],   places=2)
        assert_almost_equal(truth[i], estimate[5-i][1], places=2)

