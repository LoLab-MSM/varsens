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

obj_names = ['emBid', 'ecPARP', 'e2']

outdir = '/Users/lopezlab/temp/EARM/'
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
param_dict = load_params("/Users/lopezlab/git/earm/EARM_2_0_M1a_fitted_params.txt")
for param in model.parameters:
	if param.name in param_dict:
		param.value = param_dict[param.name]

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
solver = Solver(model, tspan, integrator='lsoda', rtol=1e-12, atol=1e-12, nsteps=20000, verbose=True)
#solver = Solver(model, tspan, integrator='vode', with_jacobian=True, atol=1e-5, rtol=1e-5, nsteps=20000)

# from pysb.generator.bng import BngGenerator
# print BngGenerator(model).get_content()
# quit()

# solver.run()
# 
# plt.figure('mBid')
# #
# upper_ICRP = np.array([min(1.,exp_data['norm_ICRP'][i]+np.sqrt(exp_data['nrm_var_ICRP'][i])) for i in range(len(exp_data['norm_ICRP']))])
# lower_ICRP = np.array([max(0.,exp_data['norm_ICRP'][i]-np.sqrt(exp_data['nrm_var_ICRP'][i])) for i in range(len(exp_data['norm_ICRP']))])
# #
# plt.plot(tspan,solver.yobs['mBid'], 'r', label='mBid', linewidth=3)
# plt.plot(tspan,exp_data['norm_ICRP']*model.parameters['Bid_0'].value, 'b', label='norm_ICRP', linewidth=3)
# plt.plot(tspan,upper_ICRP*model.parameters['Bid_0'].value, 'b--', linewidth=2)
# plt.plot(tspan,lower_ICRP*model.parameters['Bid_0'].value, 'b--', linewidth=2)
# plt.legend(loc='upper left')
# # plt.yscale('log')
# 
# plt.figure('cPARP')
# #
# upper_ECRP = np.array([min(1.,exp_data['norm_ECRP'][i]+np.sqrt(exp_data['nrm_var_ECRP'][i])) for i in range(len(exp_data['norm_ICRP']))])
# lower_ECRP = np.array([max(0.,exp_data['norm_ECRP'][i]-np.sqrt(exp_data['nrm_var_ECRP'][i])) for i in range(len(exp_data['norm_ICRP']))])
# #
# plt.plot(tspan,solver.yobs['cPARP'], 'r', label='cPARP', linewidth=3)
# plt.plot(tspan,exp_data['norm_ECRP']*model.parameters['PARP_0'].value, 'b', label='norm_ECRP', linewidth=3)
# plt.plot(tspan,upper_ECRP*model.parameters['PARP_0'].value, 'b--', linewidth=2)
# plt.plot(tspan,lower_ECRP*model.parameters['PARP_0'].value, 'b--', linewidth=2)
# plt.legend(loc='upper left')
# # plt.yscale('log')
# 
# plt.figure('aSmac')
# plt.plot(tspan,solver.yobs['aSmac'], 'r', label='aSmac', linewidth=3)
# plt.plot(tspan,exp_data['IMSRP']*model.parameters['Smac_0'].value, 'b', label='IMSRP', linewidth=3)
# plt.legend(loc='upper left')
# # plt.yscale('log')
# 
# plt.show()
	
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
	# Get yfinal, the last element from the trajectory
	yfinal = ysim_momp[-1]
	# Build a vector of the 3 variables to fit
	momp_sim = [td, ts, yfinal]
	# Perform chi-squared calculation against mean and variance vectors
	e2 = np.sum((momp_data - momp_sim) ** 2 / (2 * momp_var)) / 3
	return [emBid, ecPARP, e2]


N_SAMPLES = range(10,100,10) + range(100,501,50)
solver.verbose = False

for n_samples in N_SAMPLES:
	
	sample = Sample(len(model.parameters_rules()), n_samples, lambda x: scale.linear(x, lower_bound=0.1*ref, upper_bound=10*ref), verbose=True)
	objective = Objective(len(model.parameters_rules()), n_samples, sample, objective_func, verbose=True)
	v = Varsens(objective, verbose=True)
 	
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

