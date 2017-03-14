varsens
=======

Python Variance Based Sensitivity Analysis by the method of Saltelli
--------------------------------------------------------------------

This package is to provide in Python a model independent method (i.e. the user provides the objective) function of doing Variance Based Sensitivity Analysis. It depends on the [ghalton](https://github.com/fmder/ghalton) package and [numpy](http://www.numpy.org) being installed. The code is based on Andrea Saltelli, "[Making best use of model evaluations to compute sensitivity indices](http://www.sciencedirect.com/science/article/pii/S0010465502002801)", Computer Physics Communications 145 (2002) 280-297.

Variance Based Sensitivity Analysis is a robust method of performing sensitivity analysis on an objection function. The Saltelli method is also very efficient in the number of points required, this is accomplished by using a low discrepancy sequence to explore the parameter space. By default the library uses a Halton sequence which is one of the best low discrepancy sequence at present. One is free to use Sobol or Latin Hypercube methods--but be aware that the Saltelli method splits the space into two samples and if the corresponding parameters chosen between these two are correlated the method will not work. A simple work around is to shuffle the results before feeding them into the algorithm, as low discrepancy sequences have some unusual correlation structures if split like this. One could use a Monte Carlo sample, but that greatly reduces efficiency and is strongly not recommended.

One advantage of this method is that it not only computes the sensitivity due to each parameter (1st order), but also those of cooperative effects of combinations of the parameters. Each possible pair, and each possible group of k-2 factors, as well as total of all terms involving a factor. A wealth of robust information with a minimum number of objective function executions.

It is best to use the library to generate the sample space. The algorithm uses a low discrepancy sequence for creating 2 sets of samples, f and f'. A low discrepancy sequence maximizes information returned with each point, however it has structure and is not random. The later computations are heavily biased by any correlation between f and f', so they must be randomly shuffled. Then once we have f and f' uncorrelated, it generates additional samples that are permutations between f and f' to probe the interaction effects of parameters in the function. For a given requested n samples and k parameters the resulting sample size is 2n(1+k).

Samples can be exported and broken into batches for batch processing.

This work is licensed under the Creative Commons Attribution-NonCommercial 3.0 Unported License. To view a copy of this license, visit the included [file](LICENSE), the [CC Website](http://creativecommons.org/licenses/by-nc/3.0/) or send a letter to Creative Commons, 444 Castro Street, Suite 900, Mountain View, California, 94041, USA.

Installing
----------

First install required package [ghalton](https://pypi.python.org/pypi/ghalton).

Next install this library.

    $ python setup.py install

Short Example
-----------------

    from varsens    import *
    import numpy
    def gi_function(xi, ai): return (numpy.abs(4.0*xi-2.0)+ai) / (1.0+ai)
    def g_function(x, a): return numpy.prod([gi_function(xi, a[i]) for i,xi in enumerate(x)])
    def g_scaling(x): return x
    def g_objective(x): return g_function(x, [0, 0.5, 3, 9, 99, 99])
    v = Varsens(g_objective, g_scaling, 6, 1024, verbose=False)
    v.var_y 
    v.sens 
    v.sens_t

Please read the Saltelli paper for interpretation of the results.

Authors
-------

Shawn Garbett <shawn.garbett@Vanderbilt.edu>

Carlos Lopez <c.lopez@Vanderbilt.edu>

Alexander Lubbock <alex.lubbock@vanderbilt.edu>
