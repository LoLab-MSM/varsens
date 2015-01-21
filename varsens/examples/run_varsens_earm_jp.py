# -*- coding: utf-8 -*-
"""
Created on Fri Jul 11 17:27:41 2014

@author: pinojc
"""

#!/usr/bin/env python

from varsens import *
from earm.lopez_embedded import model
from pysb.integrate import Solver
from pysb.util import load_params
import os
import copy
import pickle
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import datetime
from multiprocessing import Pool, Value, Queue
import multiprocessing
import multiprocessing as mp
import math
obj_names = ['emBid', 'ecPARP', 'e2']



# observables = ['mBid', 'aSmac' , 'cPARP']

# Best fit parameters
param_dict = load_params("/home/pinojc/Projects/earm/EARM_2_0_M1a_fitted_params.txt")
for param in model.parameters:
    if param.name in param_dict:
        param.value = param_dict[param.name]

# Load experimental data file
exp_data = np.genfromtxt('/home/pinojc/Projects/earm/xpdata/forfits/EC-RP_IMS-RP_IC-RP_data_for_models.csv', delimiter=',', names=True)

# # Build time points for the integrator, using the same time scale as the
# # experimental data but with greater resolution to help the integrator converge.
# ntimes = len(exp_data['Time'])
# # Factor by which to increase time resolution
# tmul = 20
# # Do the sampling such that the original experimental timepoints can be
# # extracted with a slice expression instead of requiring interpolation.
# tspan = np.linspace(0.0, exp_data['Time'][-1], (ntimes-1) * tmul + 1)
# #tspan = np.linspace(exp_data['Time'][0], exp_data['Time'][-1],
# #                   (ntimes-1) * tmul + 1)

tspan = exp_data['Time']


# Initialize solver object
#solver = Solver(model, tspan, integrator='lsoda', rtol=1e-6, atol=1e-6, nsteps=20000)
solver = Solver(model, tspan, integrator='vode', with_jacobian=True, atol=1e-5, rtol=1e-5, nsteps=20000)


# Determine IDs for rate parameters, original values for all parameters to overlay, and reference values for scaling.


k_ids = [i for i,p in enumerate(model.parameters) if p in model.parameters_rules()]
k_ids_names = [p for i,p in enumerate(model.parameters) if p in model.parameters_rules()]
par_vals = np.array([p.value for p in model.parameters])

ref = np.array([p.value for p in model.parameters_rules()])
for i in range(len(k_ids_names)):
    print i,k_ids_names[i]
#quit()
# Sample takes the parameters: k or num of parameters, n or num of samples, scaling, verbose and loadFile
# =====
# Run to generate samples
# sample = Sample(len(model.parameters_rules()), 7500, lambda x: scale.magnitude(x, reference=ref, orders=1.0), verbose=True)
# n_samples = 100 #7500
# sample = Sample(len(model.parameters_rules()), n_samples, lambda x: scale.linear(x, lower_bound=0.1*ref, upper_bound=10*ref), verbose=True)

# List of model observables and corresponding data file columns for point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [model.parameters['Bid_0'].value,model.parameters['PARP_0'].value]

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory) from Spencer et al., 2009
momp_obs_total = model.parameters['Smac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])
counter = None

def objective_func(params):
    par_vals[k_ids] = params
    solver.run(par_vals)

    # Calculate error for point-by-point trajectory comparisons
    for obs_name, data_name, var_name, obs_total in zip(obs_names, data_names, var_names, obs_totals):
        # Get model observable trajectory (this is the slice expression mentioned above in the comment for tspan)
        ysim = solver.yobs[obs_name] #[::tmul]
        # Normalize it to 0-1
        ysim_norm = ysim / obs_total
        # Get experimental measurement and variance
        ydata = exp_data[data_name]
        yvar = exp_data[var_name]
        # Compute error between simulation and experiment (chi-squared)
        if obs_name == 'mBid':
            emBid = np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)
        elif obs_name == 'cPARP':
            ecPARP = np.sum((ydata - ysim_norm) ** 2 / (2 * yvar)) / len(ydata)

    # Calculate error for Td, Ts, and final value for IMS-RP reporter
    # =====
    # Normalize trajectory
    ysim_momp = solver.yobs[momp_obs]
    ysim_momp_norm = ysim_momp / np.nanmax(ysim_momp)
    # Build a spline to interpolate it
    st, sc, sk = scipy.interpolate.splrep(solver.tspan, ysim_momp_norm)
    # Use root-finding to find the point where trajectory reaches 10% and 90%
    t10 = scipy.interpolate.sproot((st, sc-0.10, sk))[0]
    t90 = scipy.interpolate.sproot((st, sc-0.90, sk))[0]
    # Calculate Td as the mean of these times
    td = (t10 + t90) / 2
    # Calculate Ts as their difference
    ts = t90 - t10
    # Get yfinal, the last element from the trajectory    yfinal = ysim_momp[-1]
    # Build a vector of the 3 variables to fit
    momp_sim = [td, ts, yfinal]
    # Perform chi-squared calculation against mean and variance vectors
    e2 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3
    etotal = np.sum(emBid, ecPARP, e2)
    return [emBid, ecPARP, e2,etotal]


solver.verbose = False
#old code, keeping in case it is better for multiple nodes
def mp_obj(sample, nprocs):
    def worker(sample1,start, out_q):

        outdict = {}
        count = start
        for n in xrange(0,len(sample1)):
            #print count
            outdict[count] = objective_func(sample[count])
            count+=1
        out_q.put(outdict)

    # Each process will get 'chunksize' nums and a queue to put his out
    # dict into
    out_q = Queue()
    chunksize = int(math.ceil(len(sample) / float(nprocs)))
    procs = []

    for i in range(nprocs):
        print "i",i

        p = multiprocessing.Process(
                target=worker,
                args=(sample[chunksize * i:chunksize * (i + 1)],chunksize * i,
                      out_q))
        procs.append(p)
        p.start()

    # Collect all results into a single result dict.
    resultdict = {}
    for i in range(nprocs):
        resultdict.update(out_q.get())


    # Wait for all worker processes to finish
    for p in procs:
        p.join()

    return resultdict

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
    n_samples = 10
    sample = Sample(len(k_ids), n_samples, lambda x: scale.linear(x, lower_bound=0.5*ref, upper_bound=5*ref), verbose=True)

    #obj_values =np.zeros(2*sample.n*(1+sample.k))
    sample = sample.flat()
    #plt.errorbar(np.arange(0,len(np.average(sample,axis=0))),np.average(sample,axis=0),yerr=np.std(sample,axis=0))
    #plt.semilogy()
    #plt.plot(ref)
    m = mp.Manager()
    #sample = m.dict(sample)
    obj_values = m.dict()
    p = mp.Pool(4,initializer = init, initargs=(sample,obj_values))
    allblocks =range(len(sample))
    p.imap_unordered(OBJ,allblocks)
    p.close()
    p.join()
   #print obj_values.values()
    #np.savetxt(str(n_samples)+'_objective_values.txt',np.asarray(obj_values))
    objective = Objective(len(k_ids), n_samples, objective_vals=np.asarray(obj_values))
    v = Varsens(objective,verbose=True)
    print 'time of '+str(np.shape(sample)[0])+' calculations '+ str(time.time() - start)
    #np.savetxt(str(n_samples)+'_1_2order_sens_50per.txt',v.sens)
    #np.savetxt(str(n_samples)+'_1_2order_sens_t_50per.txt',v.sens_t)
    #np.savetxt(str(n_samples)+'_1_2order_sens_2_embid_50per.txt',v.sens_2[:,0,:,0])
    #np.savetxt(str(n_samples)+'_1_2order_sens_2_ecPARP_50per.txt',v.sens_2[:,1,:,1])
    #np.savetxt(str(n_samples)+'_1_2order_sens_2_cSMAC_50per.txt',v.sens_2[:,2,:,2])

    #obj_names = ['emBid', 'ecPARP', 'e2']
    #sample1 = Sample(len(k_ids), n_samples, lambda x: scale.linear(x, lower_bound=0.1*ref, upper_bound=10*ref), verbose=True)
    #objective1 = Objective(len(k_ids), n_samples,sample1, objective_func)
    #v1 = Varsens(objective,verbose=True)
    #plt.plot(v.sens_t)
    #plt.legend(('mBid', 'cPARP','aSmac'),loc=0)
    #print sum(v.sens_t)
    #plt.plot(v1.sens_t)
    #plt.legend(('mBid', 'cPARP','aSmac'),loc=0)
    #print sum(v1.sens_t)
#if __name__ == '__main__':
#
#    import time
#    start = time.time()
#    n_samples = 100
#    sample = Sample(len(k_ids), n_samples, lambda x: scale.linear(x, lower_bound=0.8*ref, upper_bound=1.2*ref), verbose=True)
#    sample = sample.flat()
#
#    #pool=Pool(4,initializer=init,initargs=(counter,))
#    #counter=Value('i',0)
#
#
#    lock = multiprocessing.Lock()
#    obj_vals = mp_obj(sample,8).values()
#    #OJ=obj_vals.values()
#
#    #obj_vals = []
#    #obj_vals = pool.map(objective_func,sample)
#    #pool.close()
#    objective = Objective(len(k_ids), n_samples, objective_vals=np.asarray(obj_vals))
#    #
#    v = Varsens(objective,verbose=True)
#    #sample = Sample(len(k_ids), n_samples, lambda x: scale.linear(x, lower_bound=0.1*ref, upper_bound=10*ref), verbose=True)
#    #objective1 = Objective(len(k_ids), n_samples, sample, objective_func, verbose=True)
#
#    #v1 = Varsens(objective1,verbose=True)
#    print 'time of '+str(np.shape(sample)[0])+' calculations '+ str(time.time() - start)
#
#    np.savetxt(str(n_samples)+'_1sens_t.txt',v.sens_t)
#    np.savetxt(str(n_samples)+'_1sens.txt',v.sens)
#    plt.plot(v.sens)
#    plt.legend(('mBid', 'cPARP','aSmac'),loc=0)
#    plt.savefig(str(n_samples)+'_1sens.png')
#    plt.show()
#    plt.clf()
#    plt.plot(v.sens_t)
#    plt.legend(('mBid', 'cPARP','aSmac'),loc=0)
#    plt.savefig(str(n_samples)+'_1senT.png')
#    #plt.show()
#    plt.clf()
#    plt.imshow(v.sens.T,aspect=15,interpolation='none')
#    plt.colorbar()
#    plt.savefig(str(n_samples)+'_1heatmap.png')
#    #plt.show()
#    plt.clf()
#
#    for i in range(0,3):
#        plt.imshow(v.sens_2[:,i,:,i],interpolation='none')
#        plt.colorbar()
#        plt.savefig(str(n_samples)+'_'+str(i)+'.png')
#        plt.clf()

#np.savetxt(str(n_samples)+'_sens_2.txt',v.sens_2)

#for n in range(v.sens.shape[1]):
#    OUTPUT[obj_names[n]+'_sens'].write(str(n_samples))
#    OUTPUT[obj_names[n]+'_sens_t'].write(str(n_samples))
#    for i in range(len(v.sens)):
#        OUTPUT[obj_names[n]+'_sens'].write("\t"+str(v.sens[i][n]))
#        OUTPUT[obj_names[n]+'_sens_t'].write("\t"+str(v.sens_t[i][n]))
#        OUTPUT[obj_names[n]+'_sens'].write("\n")
#        OUTPUT[obj_names[n]+'_sens_t'].write("\n")
#        sens_2 = v.sens_2[:,n,:,n] # 2-D matrix
#        OUTPUT[obj_names[n]+'_sens_2'].write(str(n_samples))
#        for i in range(len(sens_2)):
#            for j in range(i,len(sens_2[i])):
#                sens_2[i][j] -= (v.sens[i][n] + v.sens[j][n])
#                sens_2[j][i] = sens_2[i][j]
#                OUTPUT[obj_names[n]+'_sens_2'].write("\t"+str(sum(sens_2[i]) - sens_2[i][i]))
#            OUTPUT[obj_names[n]+'_sens_2'].write("\n")
#
#    # flush output
#    for file in OUTPUT.values(): file.flush()
#for file in OUTPUT.values(): file.close()
