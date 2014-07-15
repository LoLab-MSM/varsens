#!/usr/bin/env python

from varsens import *
from earm.lopez_embedded import model
from pysb.integrate import Solver
from pysb.util import load_params
from pysb.tools.cupSODA import *
import os
import copy
import pickle
import numpy as np
import scipy.interpolate
import matplotlib.pyplot as plt
import datetime

##### Set paths
outdir = '/Users/lopezlab/temp/EARM/'
set_cupSODA_path("/Users/lopezlab/cupSODA")
#####

par_names = [p.name for p in model.parameters_rules()]
par_dict  = {name : index for index,name in enumerate(par_names)}

obj_names = ['emBid', 'ecPARP', 'e2']

outfiles = {}
for name in obj_names:
	outfiles[name+'_sens']    = os.path.join(outdir,name+"_sens_scipy.txt")
	outfiles[name+'_sens_t']  = os.path.join(outdir,name+"_sens_t_scipy.txt")
	outfiles[name+'_sens_2']  = os.path.join(outdir,name+"_sens_2_scipy.txt")
OUTPUT = {key : open(file, 'a') for key,file in outfiles.items()}
for file in OUTPUT.values():
	file.write("-----"+str(datetime.datetime.now())+"-----\n")
	file.write("n")
	for i in range(len(model.parameters_rules())):
    		file.write("\t"+"p_"+str(i))
	file.write("\n")

# Best fit parameters
params_fitted = load_params("/Users/lopezlab/git/earm/EARM_2_0_M1a_fitted_params.txt")
for p in model.parameters:
	if p.name in params_fitted:
		p.value = params_fitted[p.name]

# Load experimental data file
exp_data = np.genfromtxt('/Users/lopezlab/git/earm/xpdata/forfits/EC-RP_IMS-RP_IC-RP_data_for_models.csv', delimiter=',', names=True)

# Time points (same as for experiments)
tspan = exp_data['Time']

# plt.figure()
# plt.plot(tspan,exp_data['ECRP'],linewidth=3,label='ECRP')
# plt.plot(tspan,exp_data['norm_ECRP'],linewidth=3,label='norm_ECRP')
# plt.legend(loc='lower right')
#  
# plt.figure()
# plt.plot(tspan,exp_data['ICRP'],linewidth=3,label='ICRP')
# plt.plot(tspan,exp_data['norm_ICRP'],linewidth=3,label='norm_ICRP')
# plt.legend(loc='lower right')

# Initialize solver object
solver = cupSODA(model, tspan, atol=1e-12, rtol=1e-6, verbose=True)
	
# Determine IDs for rate parameters, original values for all parameters to overlay, and reference values for scaling.
k_ids = [i for i,p in enumerate(model.parameters) if p in model.parameters_rules()]
par_vals = np.array([p.value for p in model.parameters])
ref = np.array([p.value for p in model.parameters_rules()])

# List of model observables and corresponding data file columns for point-by-point fitting
obs_names = ['mBid', 'cPARP']
data_names = ['norm_ICRP', 'norm_ECRP']
var_names = ['nrm_var_ICRP', 'nrm_var_ECRP']
# Total starting amounts of proteins in obs_names, for normalizing simulations
obs_totals = [model.parameters['Bid_0'].value,
			  model.parameters['PARP_0'].value]

# Model observable corresponding to the IMS-RP reporter (MOMP timing)
momp_obs = 'aSmac'
# Mean and variance of Td (delay time) and Ts (switching time) of MOMP, and
# yfinal (the last value of the IMS-RP trajectory) from Spencer et al., 2009
momp_obs_total = model.parameters['Smac_0'].value
momp_data = np.array([9810.0, 180.0, momp_obs_total])
momp_var = np.array([7245000.0, 3600.0, 1e4])

# def objective_func(params):
def objective_func(yobs):
	
# 	par_vals[k_ids] = params
# 	solver.run(par_vals)
	
	# Calculate error for point-by-point trajectory comparisons
	for obs_name, data_name, var_name, obs_total in zip(obs_names, data_names, var_names, obs_totals):
		# Get model observable trajectory (this is the slice expression mentioned above in the comment for tspan)
# 		ysim = solver.yobs[obs_name] #[::tmul]
		ysim = yobs[obs_name]
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
# 	ysim_momp = solver.yobs[momp_obs]
	ysim_momp = yobs[momp_obs]
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
	# Get yfinal, the last element from the trajectory
	yfinal = ysim_momp[-1]
	# Build a vector of the 3 variables to fit
	momp_sim = [td, ts, yfinal]
	# Perform chi-squared calculation against mean and variance vectors
	e2 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3
	
	return [emBid, ecPARP, e2]

n_iter = 2
n_start = 2
N_SAMPLES = [n_start*iter for iter in range(1,n_iter+1)]

threads_per_block = 32
solver.verbose = False

for n_samples in N_SAMPLES:
	
	total_sims = 2*n_samples*(1+len(par_names))
	n_blocks = min(2354, int(round(1.*total_sims/threads_per_block)))
	sims_per_batch = n_blocks*threads_per_block
	n_batches = max( 2*n_samples*(1+len(par_names)), sims_per_batch ) / sims_per_batch
	
	#####
# 	print "n_samples:", n_samples
# 	print "total_sims:", total_sims
# 	print "n_blocks:", n_blocks
# 	print "sims_per_batch:", sims_per_batch
# 	print "n_batches:", n_batches
# 	print "n_batches*sims_per_batch:", n_batches*sims_per_batch, "("+str(total_sims)+")"
# 	print "----------------"
# 	n_samples *= 2
# 	continue
	#####

	sample = Sample(len(model.parameters_rules()), n_samples, lambda x: scale.linear(x, lower_bound=0.1*ref, upper_bound=10*ref))
	sample_flat = sample.flat()
	obj_vals = np.zeros((total_sims, len(obs_names)))
	
	for batch in range(n_batches):
		
		start = batch*sims_per_batch
		end = min(start+sims_per_batch, total_sims)
		sample_batch = sample_flat[start:end]
		
		# Rate constants
		c_matrix = np.zeros((len(sample_batch), len(model.reactions)))
		rate_args = []
		for rxn in model.reactions:
			rate_args.append([arg for arg in rxn['rate'].args if not re.match("_*s",str(arg))])
		output = 0.01*len(sample_batch)
		output = int(output) if output > 1 else 1
		for i in range(len(sample_batch)):
			if i % output == 0: print str(int(round(100.*i/len(sample_batch))))+"%"
			for j in range(len(model.reactions)):
				rate = 1.0
				for r in rate_args[j]:
					x = str(r)
					if x in par_dict.keys():
						rate *= sample_batch[i][par_dict[x]]
					else:
						rate *= float(x)
				c_matrix[i][j] = rate
		
		# Initial concentrations
		MX_0 = np.zeros((len(sample_batch),len(model.species)))
        	for i in range(len(model.initial_conditions)):
        		for j in range(len(model.species)):
        			if str(model.initial_conditions[i][0]) == str(model.species[j]): # The ComplexPattern objects are not the same, even though they refer to the same species (ask about this)
        				x = model.initial_conditions[i][1]
        				if (x.name in par_dict.keys()):
        					MX_0[:,j] = sample_batch[:,par_dict[x.name]]
        				else:
        					MX_0[:,j] = [x.value for i in range(len(sample_batch))]
        				break
        
        solver.run(c_matrix, MX_0, outdir=os.path.join(outdir,'NSAMPLES_'+str(n_samples))) #obs_species_only=False, load_conc_data=False)
        os.rename(os.path.join(solver.outdir,"__CUPSODA_FILES"), os.path.join(solver.outdir,"__CUPSODA_FILES_"+str(batch)))
        
        for i in range(end-start):
        		obj_vals[i] = objective_fun(solver.yobs[i])
	
	objective = Objective(len(par_vals), n_samples, objective_vals=obj_vals)
	v = Varsens(objective)
	
	for n in range(v.sens.shape[1]):
		
		# sens & sens_t
		OUTPUT[obj_names[n]+'_sens'].write(str(n_samples))
		OUTPUT[obj_names[n]+'_sens_t'].write(str(n_samples))
		for i in range(len(v.sens)):
			OUTPUT[obj_names[n]+'_sens'].write("\t"+str(v.sens[i][n]))
			OUTPUT[obj_names[n]+'_sens_t'].write("\t"+str(v.sens_t[i][n]))
		OUTPUT[obj_names[n]+'_sens'].write("\n")
		OUTPUT[obj_names[n]+'_sens_t'].write("\n")
		
		# sens_2
		sens_2 = v.sens_2[:,n,:,n] # 2-D matrix
		OUTPUT[obj_names[n]+'_sens_2'].write(str(n_samples))
		for i in range(len(sens_2)):
			for j in range(i,len(sens_2[i])):
				sens_2[i][j] -= (v.sens[i][n] + v.sens[j][n])
				sens_2[j][i] = sens_2[i][j]
			OUTPUT[obj_names[n]+'_sens_2'].write("\t"+str(sum(sens_2[i]) - sens_2[i][i]))
		OUTPUT[obj_names[n]+'_sens_2'].write("\n")
		
	# flush output
	for file in OUTPUT.values(): file.flush()
	
for file in OUTPUT.values(): file.close()

