from erbb_exec import model
import pickle
from pysb.integrate import Solver
import numpy as np
import copy
from varsens import *
import os

def extract_records(recarray, names):
    """Convert a record-type array and list of names into a float array"""
    return np.vstack([recarray[name] for name in names]).T

t = np.linspace(0,7200, num=7200)
observables = ['obsAKTPP', 'obsErbB1_P_CE', 'obsERKPP']

#Sub best-fit parameters
with open('/home/shockle/egfr/egfr/calibration_A431_EGF_fittedparams_unnorm_splined_fromoriginal_nokf_justerbbparams_fixedmodel_refitted_2_bestvals', 'rb') as handle:
     fittedparams = pickle.loads(handle.read())

for i in range(len(model.parameters)):
     if model.parameters[i].name in fittedparams:
         model.parameters[i].value = fittedparams[model.parameters[i].name]
         

solver = Solver(model, t, integrator='vode', with_jacobian=True, atol=1e-6, rtol=1e-6, nsteps=10000)
solver.run()

expdata = np.load('/home/shockle/egfr/egfr/experimental_data_A431_highEGF_unnorm.npy')
expdata_var = np.load('/home/shockle/egfr/egfr/experimental_data_var_A431_highEGF_unnorm.npy')
exptimepts = np.array([0, 149, 299, 449, 599, 899, 1799, 2699, 3599, 7199])

#Determine IDs for rate parameters, original values for all parameters to overlay, and reference values for scaling.
kids = [i for i, p in enumerate(model.parameters) if p in model.parameters_rules()]
vals = np.array([p.value for p in model.parameters])
reference = np.array([p.value for p in model.parameters_rules()])

def obj_func(params):
    vals[kids] = params
    solver.run(vals)
    sim = copy.copy(solver.yobs)
    sim_array = extract_records(sim,observables)
    sim_slice = sim_array[exptimepts]
    obj_AKT = np.sum((expdata[:,0] - sim_slice[:,0]) ** 2 / (2 * expdata_var[:,0] ** 2))
    obj_ErbB1 = np.sum((expdata[:,1] - sim_slice[:,1]) ** 2 / (2 * expdata_var[:,1] ** 2))
    obj_ERK = np.sum((expdata[:,2] - sim_slice[:,2]) ** 2 / (2 * expdata_var[:,2] ** 2))
    return np.array([obj_AKT, obj_ErbB1, obj_ERK])

#n*(2*k+2) - for erbb_exec model = 7500 * (2*215 + 2) = 3240000
#Run to generate samples
sample = Sample(len(model.parameters_rules()), 7500, lambda x: scale.magnitude(x, reference=reference, orders=1.0), verbose=True)
os.chdir('/scratch/shockle/varsens/samples')
sample.export(prefix='erbb_sample_', postfix='', blocksize=2160)

#Run below in parallel to generate objective functions (with input samplefile from command line)
#samplegroup = Sample(len(model.parameters_rules()), 5, lambda x: x, verbose=True, loadFile=samplefile)
#objective = Objective(len(model.parameters_rules()), 5, samplegroup, obj_func, verbose=True)
#objective.export('erbb_objective_', '', block)
