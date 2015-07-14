========
Tutorial
========

Introduction
============

This tutorial will walk you through a basic sensitivity analysis.
   
First steps
===========

The first step is truly understanding your model, and its outputs. Sensitivity analysis ranks parameters of the model
versus a given output, i.e. how much a change in a parameter will effect the output. For this example, we're going to
use Sobol's g-function.

.. code:: python

    def gi_function(xi, ai):
        return (numpy.abs(4.0*xi-2.0)+ai) / (1.0+ai)

    def g_function(x, a):
        return numpy.prod([gi_function(xi, a[i]) for i,xi in enumerate(x)])

    model = [0, 0.5, 3, 9, 99, 99]

    def g_objective(x): return g_function(x, model)

This bit of python defines a model, the g-function, and an objective function--the output of the model that it is
desired to determine the effects of the parameters passed in through x. The model variable basically ranks the
dimensions, a 0 having the most effect and a 99 having the least effect on the output.

At this point one could use this library to get the estimate. Just specify the objective function to measure, a sample
space transform (the identity function in this case), the number of dimensions, the number of samples, and the method of
 Saltelli is applied.

.. code:: python

    from varsens import *
    v = Varsens(g_objective, lambda x: x, len(model), 50000)

that would miss all the details and options that are hidden underneath. Especially when one is interested in batch
running jobs for complex models and such. So we're going to do a longer slower example to achieve the same thing.
First up, is generating a sample.

.. code:: python

    s = Sample(len(model), 50000, lambda x: x)

This returns a sample object full of the sample space to analyze. While the number of dimensions and samples is
straightforward, the scaling function is the only additional thing to understand. The sample space generated is list of
vectors that are all in the range [0..1]. The scaling function takes one of these vectors and scales it into the
parameter space of the model. There is an additional scale library that contains 4 commons methods of scaling
parameters. Next each of these vectors will be provided to the user objective function to evaluate a model at that point
 in the sample space.

.. code:: python 
	
    o = Objective(len(model), 50000, s, g_objective)

Now the objective function has been run against the sample space and the results stored in the objective object.
This operation is batchable. One could for example use the export function on a sample space, and create csv batches of
samples to run. Then the objective function could be run on these, to generate a series of matching objectives in a csv
file as well. This would get loaded into the objective using a load function.

.. code:: python

    v = Varsens(o)

At this point the variance based sensitivity analysis is done on the model. The contents of v.sens, v.sens_t, v.sens_2,
v.sens_2t, v.var_y and v.E_2 contain the information desired. For an interpretation of these values, please refer to
Saltelli's papers on the topic.