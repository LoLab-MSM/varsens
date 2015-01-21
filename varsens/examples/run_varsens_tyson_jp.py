from varsens import *
from pysb.examples.tyson_oscillator import model
from pysb.integrate import Solver
from pysb.integrate import *
import os
import copy
import pickle
import numpy as np
import matplotlib.pyplot as plt
import datetime
from multiprocessing import Pool, Value, Queue
import multiprocessing
import multiprocessing as mp
import math


par_names = ['k1', 'k3', 'k4', 'kp4', 'k6', 'k8', 'k9']

outdir = '/home/pinojc/git/varsens/varsens/examples/Tyson_output/'


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
percent = 10
scaling = [par_vals-int(percent)/100.*par_vals, par_vals+int(percent)/100.*par_vals] # 20% around values

def scale(points):
    return points * (scaling[1] - scaling[0]) + scaling[0]

def objective_func(params):
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


solver.verbose = False

def init(sample,dictionary):
    global Sample
    global Dictionary
    Sample,Dictionary = sample,dictionary

def OBJ(block):
    #print block
    obj_values[block]=objective_func(sample[block])

if __name__ == '__main__':
    import time
    start = time.time()
    n_samples = 10000
    sample = Sample(len(par_vals), n_samples,  scale, verbose=True)
    sample = sample.flat()
    m = mp.Manager()
    obj_values = m.dict()
    p = mp.Pool(4,initializer = init, initargs=(sample,obj_values))
    allblocks =range(len(sample))
    p.imap_unordered(OBJ,allblocks)
    p.close()
    p.join()
    tmp=np.asarray(obj_values.values()).reshape((len(obj_values),1))
    #np.savetxt(str(n_samples)+'_objective_values.txt',np.asarray(obj_values))
    objective = Objective(len(par_vals), n_samples, objective_vals=tmp)
    v = Varsens(objective,verbose=True)
    print 'time of '+str(np.shape(sample)[0])+' calculations '+ str(time.time() - start)
    np.savetxt(outdir+'sens_nsample_'+str(n_samples)+'_percent_'+str(percent)+'.txt',v.sens)
    np.savetxt(outdir+'sens_t_nsample_'+str(n_samples)+'_percent_'+str(percent)+'.txt',v.sens_t)
    np.savetxt(outdir+'sens_2_nsample_'+str(n_samples)+'_percent_'+str(percent)+'.txt',v.sens_2[:,0,:,0])

    
