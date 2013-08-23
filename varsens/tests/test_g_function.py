from varsens    import *
from nose.tools import *
import numpy
import itertools

########################################################
# Sobel's g-function for testing
# Eq (30)
def gi_function(xi, ai):
    return (numpy.abs(4.0*xi-2.0)+ai) / (1.0+ai)

# Eq (29)
def g_function(x, a):
    return numpy.prod([gi_function(xi, a[i]) for i,xi in enumerate(x)])

model = [0, 0.5, 3, 9, 99, 99]

def g_objective(x): return g_function(x, model)

def g_truth(model):
    return 1.0/(3.0*((numpy.array(model) + 1.0)**2.0))

def g_truth_2(model, i, j):
    x = g_truth(model)
    return x[i]+x[j]+x[i]*x[j]

def g_truth_vnc(model, l):
    x = g_truth(model)
    result = 0.0
    k = len(model)
    others = range(k)
    for i in l: others.remove(i)
    for j in range(k):
        for m in itertools.combinations(others, j+1):
            result += numpy.prod(x[numpy.array(m)])
    return result

def g_truth_t(model, i):
    x = g_truth(model)
    return x[i]*(1.0+g_truth_vnc(model, [i]))

def g_var(model):
    x = g_truth(model)
    result = 0.0
    k = len(model)
    all = range(k)
    for j in range(k):
        for m in itertools.combinations(all, j+1):
            result += numpy.prod(x[numpy.array(m)])
    return result

def test_g_function():
    # Analytical answer, Eq (34) divided by V(y), matches figure
    v = Varsens(g_objective, lambda x: x, 6, 1024*10, verbose=False)

    # Remove effect of estimating variance
    estimate    = v.sens    * v.var_y
    estimate_2  = v.sens_2  * v.var_y
    estimate_2n = v.sens_2n * v.var_y
    estimate_t  = v.sens_t  * v.var_y

    # Get the truth
    truth      = g_truth(model)

    assert_almost_equal(g_var(model), v.var_y, places=2)
    assert_almost_equal(1.0,          v.E_2,   places=2)

    for i in range(v.k):
        assert_almost_equal(truth[i],            estimate[i],   places=2)
        assert_almost_equal(g_truth_t(model, i), estimate_t[i], places=2)
        for j in range(i+1, v.k):
            assert_almost_equal(g_truth_2(model, i, j),     estimate_2[i,j],  places=2)
            assert_almost_equal(g_truth_vnc(model, [i, j]), estimate_2n[i,j], places=2)

# This tests when an objective function returns multiple values
# This combines the g_function result, with it's reversed model of results
def g_double_objective(x): return [g_function(x, model), g_function(x, model[::-1])]

def test_double_g_function():
    # Analytical answer, Eq (34) divided by V(y), matches figure
    v = Varsens(g_double_objective, lambda x: x, 6, 1024*10, verbose=False)
    estimate = v.sens * v.var_y
    print estimate
    truth    = g_truth(model)

    for i in range(v.k):
        assert_almost_equal(truth[i], estimate[i][0],   places=2)
        assert_almost_equal(truth[i], estimate[5-i][1], places=2)

