# This creates model.odes which contains the math
from pysb.bng import *
from pysb.integrate import *
from pysb.examples.tyson_oscillator import model
import numpy as np
# from pysb.tools.tropicalize import *
import matplotlib.pyplot as plt
from varsens import *
from pysb.generator.bng import BngGenerator
from pysb.tools.cupSODA import *
import filecmp
import os
import re
import datetime

par_names = ['k1', 'k3', 'k4', 'kp4', 'k6', 'k8', 'k9']

outdir = '/home/pinojc/git/varsens/varsens/examples/Tyson_output'
outfiles = {
            'ALL'    : os.path.join(outdir,"tyson_sens_ALL_scipy.txt"),
            'sens'   : os.path.join(outdir,"tyson_sens_scipy.txt"),
            'sens_t' : os.path.join(outdir,"tyson_sens_t_scipy.txt")
           }
# for file in outfiles.values():
#     if os.path.exists(file):
#         os.remove(file)
OUTPUT = {key : open(file, 'a') for key,file in outfiles.items()}
for file in OUTPUT.values():
    file.write("-----"+str(datetime.datetime.now())+"-----\n")
OUTPUT['sens'].write("n")
OUTPUT['sens_t'].write("n")
for p in par_names:
    OUTPUT['sens'].write("\t"+p)
    OUTPUT['sens_t'].write("\t"+p)
OUTPUT['sens'].write("\n")
OUTPUT['sens_t'].write("\n")

generate_equations(model,verbose=True)
# print BngGenerator(model).get_content()

t = np.linspace(0, 500, 5001)
ref = odesolve(model, t, integrator='lsoda', verbose=False)

# for i in range(len(model.observables)):
#     obs = str(model.observables[i])
#     plt.plot(t,ref[obs],label=obs)
# plt.legend(loc='lower right', shadow=True, prop={'size':10})
# plt.yscale('log')
# #plt.ylim(ymin=1e-5)
# plt.show()
# quit()

solver = Solver(model, t, integrator='lsoda', verbose=False)

par_vals = np.array([model.parameters[nm].value for nm in par_names])
scaling = [par_vals-0.2*par_vals, par_vals+0.2*par_vals] # 20% around values

def scale(points):
    return points * (scaling[1] - scaling[0]) + scaling[0]

def osc_objective(params):
    for i in range(len(params)):
        model.parameters[par_names[i]].value = params[i]
    solver.run()
    x = solver.yobs
    #####
#     for i in range(len(model.observables)):
#         obs = str(model.observables[i])
#         plt.plot(t,x[obs],label=obs)
#     plt.show()
    #####
    return sum((x['YT'] - ref['YT'])**2)

n_iter = 7
n_samples = 10

for iter in range(n_iter):
    
    sample = Sample(len(par_vals), n_samples, scale)
    objective = Objective(len(par_vals), n_samples, sample, osc_objective)
    v = Varsens(objective)

    # tyson_sens_ALL.txt
    OUTPUT['ALL'].write("n:\t" + str(n_samples) + "\n")
    OUTPUT['ALL'].write("var_y:\t" + str(v.var_y) + "\n")
    OUTPUT['ALL'].write("sens:")
    for i in numpy.hstack(v.sens):
        OUTPUT['ALL'].write("\t"+str(i))
    OUTPUT['ALL'].write("\n")
    OUTPUT['ALL'].write("sens_t:")
    for i in numpy.hstack(v.sens_t):
        OUTPUT['ALL'].write("\t"+str(i))
    OUTPUT['ALL'].write("\n")
    OUTPUT['ALL'].write("sens_2:")
    for i in range(len(v.sens_2)):
        for j in numpy.hstack(v.sens_2[i]):
            OUTPUT['ALL'].write("\t"+str(j))
        OUTPUT['ALL'].write("\n")
    OUTPUT['ALL'].write("sens_2n:")
    for i in range(len(v.sens_2n)):
        for j in numpy.hstack(v.sens_2n[i]):
            OUTPUT['ALL'].write("\t"+str(j))
        OUTPUT['ALL'].write("\n")
    OUTPUT['ALL'].write("--------------------\n")
    # tyson_sens.txt & tyson_sens_t.txt
    OUTPUT['sens'].write(str(n_samples))
    OUTPUT['sens_t'].write(str(n_samples))
    for i in range(len(par_names)):
        OUTPUT['sens'].write("\t"+str(v.sens[i]))
        OUTPUT['sens_t'].write("\t"+str(v.sens_t[i]))
    OUTPUT['sens'].write("\n")
    OUTPUT['sens_t'].write("\n")
    # flush output
    for file in OUTPUT.values(): file.flush()
     
    n_samples += 10
     
for file in OUTPUT.values(): file.close()
