from varsens    import *
from nose.tools import *
from tempfile   import *
import numpy 
import itertools
import shutil

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
    
def run_objectives(tmpdir, b):
    for i in range(b):
        result = []
        samples = numpy.loadtxt(tmpdir+("/batch-%d.csv" % (i+1)))
        for s in samples:
            result.append(g_objective(s))
        numpy.savetxt(tmpdir+("/obj-%d.csv" % (i+1)), numpy.array(result))

def load_results(tmpdir, s, b):
    o = Objective(s.k, s.n, s, None)
    o.load(tmpdir+"/obj-", ".csv", b)
    # Now compute results from batch
    v = Varsens(o, sample=s)
    return v

def test_import_export():
    tmpdir = mkdtemp()
    
    print(tmpdir)
    
    k = 6
    n = 1024
    
    # Analytical answer, Eq (34) divided by V(y), matches figure
    s = Sample(k, n, lambda x: x)
    v = Varsens(g_objective, verbose=False, sample=s)
    
    # Do a batch run, simulating ACCRE
    s.export(tmpdir+"/batch-", ".csv", 200)
    run_objectives(tmpdir, 72)
    v2 = load_results(tmpdir, s, 72)

    shutil.rmtree(tmpdir)
    
    # Test the results
    assert_almost_equal(v.var_y, v2.var_y)
    assert_almost_equal(v.E_2,   v2.E_2  )
    for i in range(k):
        assert_almost_equal(v.sens[i],    v2.sens[i]   )
        assert_almost_equal(v.sens_t[i],  v2.sens_t[i] )
        for j in range(k):
            assert_almost_equal(v.sens_2[i][j],  v2.sens_2[i][j] )
            assert_almost_equal(v.sens_2n[i][j], v2.sens_2n[i][j])



