from varsens import *
import random
import numpy
import itertools
from scipy.stats import t

########################################################
# Sobel's g-function for testing
# Eq (30)
def gi_function(xi, ai):
    return (numpy.abs(4.0*xi-2.0)+ai) / (1.0+ai)

# Eq (29)
def g_function(x, a):
    return numpy.prod([gi_function(xi, a[i]) for i,xi in enumerate(x)])

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

def g_objective(x): return g_function(x, model)

def sample_g_function(s):

    k = s.k
    n = s.n
    
    model = [random.uniform(0, 99) for i in range(k)]
    
    v = Varsens(lambda x: g_function(x, model), sample=s, verbose=False)

    # Remove effect of estimating variance
    estimate    = v.sens    * v.var_y

    # Get the truth
    truth      = g_truth(model)
    
    # Compute error of 1st order sensitivity analysis
    error = 0
    for i in range(k):
    	error += (truth[i] - estimate[i])**2
    
    #error /= k # Scale to number of dimensions
    
    return error

def bootstrap_estimate(b, n, k):
	s= Sample(k, n, lambda x: x)
	x=[sample_g_function(s) for i in range(b)]
	mu = numpy.mean(x)
	sd = numpy.std(x)
	se = sd / numpy.sqrt(b)
	lci = mu - se*t.isf(0.025, b-1)
	uci = mu + se*t.isf(0.025, b-1)
	return (n, mu, sd, lci, uci, numpy.amax(x))


x = bootstrap_estimate(100, 5, 197)
import matplotlib.pyplot as plt
plt.hist(x)
plt.show()

# Profile across n for k=197
samples = [5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240, 20480] 
b=10
#profile_n = numpy.array([(n, numpy.average(bootstrap_estimate(b, n, 197))) for n in samples])


# Doing this craziness so I can abort if it takes too long
x1  = bootstrap_estimate(b, 5,     197)
x2  = bootstrap_estimate(b, 10,    197)
x3  = bootstrap_estimate(b, 20,    197)
x4  = bootstrap_estimate(b, 40,    197)
x5  = bootstrap_estimate(b, 80,    197)
x6  = bootstrap_estimate(b, 160,   197)
x7  = bootstrap_estimate(b, 320,   197)
x8  = bootstrap_estimate(b, 640,   197)
x9  = bootstrap_estimate(b, 1280,  197)
x10 = bootstrap_estimate(b, 2560,  197)
x11 = bootstrap_estimate(b, 5120,  197)
x12 = bootstrap_estimate(b, 10240, 197)
x13 = bootstrap_estimate(b, 20480, 197)

profile_n = numpy.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])

# Raw result
from pylab import * 
errorbar(numpy.log10(profile_n[:,0]), numpy.sqrt(profile_n[:,1]), yerr=numpy.sqrt(profile_n[:,2]*t.isf(0.25,b-1) / numpy.sqrt(b)))
show()

# Rescaled to total error
from pylab import * 
errorbar(numpy.log10(profile_n[:,0]), numpy.sqrt(profile_n[:,1]*197), yerr=numpy.sqrt(profile_n[:,2]*197.0*197.0*t.isf(0.25,b-1) / numpy.sqrt(b)))
show()

from pylab import * 
plot(numpy.log10(profile_n[:,0]), numpy.sqrt(profile_n[:,1]*197))
show()


# Doing this craziness so I can abort if it takes too long
b=30
x1  = bootstrap_estimate(b, 5,     6)
x2  = bootstrap_estimate(b, 10,    6)
x3  = bootstrap_estimate(b, 20,    6)
x4  = bootstrap_estimate(b, 40,    6)
x5  = bootstrap_estimate(b, 80,    6)
x6  = bootstrap_estimate(b, 160,   6)
x7  = bootstrap_estimate(b, 320,   6)
x8  = bootstrap_estimate(b, 640,   6)
x9  = bootstrap_estimate(b, 1280,  6)
x10 = bootstrap_estimate(b, 2560,  6)
x11 = bootstrap_estimate(b, 5120,  6)
x12 = bootstrap_estimate(b, 10240, 6)
x13 = bootstrap_estimate(b, 20480, 6)

profile_n = numpy.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])

numpy.savetxt("error-profile-dim6.csv", profile_n, delimiter=",")


from pylab import * 
errorbar(profile_n[:,0],
         numpy.sqrt(profile_n[:,1]))
show()



error = log10(profile_n[:,4]) - log10(profile_n[:,3])


# LINEAR!!!! Power Law it is!
from pylab import * 
errorbar(numpy.log10(profile_n[:,0]),
         numpy.log10(numpy.sqrt(profile_n[:,1])),
         yerr=error)
show()

# Doing this craziness so I can abort if it takes too long
b=30
x1  = bootstrap_estimate(b, 5,     12)
x2  = bootstrap_estimate(b, 10,    12)
x3  = bootstrap_estimate(b, 20,    12)
x4  = bootstrap_estimate(b, 40,    12)
x5  = bootstrap_estimate(b, 80,    12)
x6  = bootstrap_estimate(b, 160,   12)
x7  = bootstrap_estimate(b, 320,   12)
x8  = bootstrap_estimate(b, 640,   12)
x9  = bootstrap_estimate(b, 1280,  12)
x10 = bootstrap_estimate(b, 2560,  12)
x11 = bootstrap_estimate(b, 5120,  12)
x12 = bootstrap_estimate(b, 10240, 12)
x13 = bootstrap_estimate(b, 20480, 12)

profile_n = numpy.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])

numpy.savetxt("error-profile-dim12.csv", profile_n, delimiter=",")

error = log10(profile_n[:,4]) - log10(profile_n[:,3])


# LINEAR!!!! Power Law it is!
from pylab import * 
errorbar(numpy.log10(profile_n[:,0]),
         numpy.log10(numpy.sqrt(profile_n[:,1])),
         yerr=error)
show()


b=30
x1  = bootstrap_estimate(b, 5,     24)
x2  = bootstrap_estimate(b, 10,    24)
x3  = bootstrap_estimate(b, 20,    24)
x4  = bootstrap_estimate(b, 40,    24)
x5  = bootstrap_estimate(b, 80,    24)
x6  = bootstrap_estimate(b, 160,   24)
x7  = bootstrap_estimate(b, 320,   24)
x8  = bootstrap_estimate(b, 640,   24)
x9  = bootstrap_estimate(b, 2480,  24)
x10 = bootstrap_estimate(b, 2560,  24)
x11 = bootstrap_estimate(b, 5240,  24)
x12 = bootstrap_estimate(b, 10240, 24)
x13 = bootstrap_estimate(b, 20480, 24)

profile_n = numpy.array([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11, x12, x13])

numpy.savetxt("error-profile-dim24.csv", profile_n, delimiter=",")

error = log10(profile_n[:,4]) - log10(profile_n[:,3])


# LINEAR!!!! Power Law it is!
from pylab import * 
errorbar(numpy.log10(profile_n[:,0]),
         numpy.log10(numpy.sqrt(profile_n[:,1])),
         yerr=error)
show()