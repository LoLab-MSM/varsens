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

outdir = '/Users/lopezlab/temp/TYSON/'
outfiles = {
            'ALL'    : os.path.join(outdir,"tyson_sens_ALL_cupSODA.txt"),
            'sens'   : os.path.join(outdir,"tyson_sens_cupSODA.txt"),
            'sens_t' : os.path.join(outdir,"tyson_sens_t_cupSODA.txt")
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
ref = odesolve(model, t, integrator='lsoda', verbose=True)

# for i in range(len(model.observables)):
#     obs = str(model.observables[i])
#     plt.plot(t,ref[obs],label=obs)
#        
# plt.legend(loc='lower right', shadow=True, prop={'size':10})
# plt.yscale('log')
# #plt.ylim(ymin=1e-5)
# plt.show()
# quit()

set_cupSODA_path("/Users/lopezlab/cupSODA")
solver = cupSODA(model, t, atol=1e-12, rtol=1e-6, verbose=True)

par_dict = {par_names[i] : i for i in range(len(par_names))}
par_vals = np.array([model.parameters[nm].value for nm in par_names])
scaling = [par_vals-0.2*par_vals, par_vals+0.2*par_vals] # 20% around values

def scale(points):
    return points * (scaling[1] - scaling[0]) + scaling[0]

n_iter = 1 #14
threads_per_block = 32
n_samples = 39

for iter in range(n_iter):
    
    total_sims = 2*n_samples*(1+len(par_names))
    n_blocks = min(624, int(round(1.*total_sims/threads_per_block)))
    sims_per_batch = n_blocks*threads_per_block
    n_batches = max( 2*n_samples*(1+len(par_names)), sims_per_batch ) / sims_per_batch
    
#     #####
#     print "n_samples:", n_samples
#     print "total_sims:", total_sims
#     print "n_blocks:", n_blocks
#     print "sims_per_batch:", sims_per_batch
#     print "n_batches:", n_batches
#     print "n_batches*sims_per_batch:", n_batches*sims_per_batch, "("+str(total_sims)+")"
#     print "----------------"
#     n_samples *= 2
#     continue
#     #####
    
    sample = Sample(len(par_vals), n_samples, scale)
#     sample.export(outdir="/Users/lopezlab/temp/TYSON", prefix="tyson_sample", blocksize=sims_per_batch)
    sample_flat = sample.flat()
    obj_vals = numpy.zeros(total_sims)
    
    for batch in range(n_batches):
        
        start = batch*sims_per_batch
        end = min(start+sims_per_batch, total_sims)
        sample_batch = sample_flat[start:end]
    
        # Rate constants
        c_matrix = np.zeros((len(sample_batch),len(model.reactions)))
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
                        rate *= sample_batch[i][par_dict[x]] # model.parameters[x].value
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
        
        x = solver.yobs
        obj_vals[start:end] = [sum((x[i]['YT'] - ref['YT'])**2) for i in range(end-start)]

    objective = Objective(len(par_vals), n_samples, objective_vals=obj_vals)
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

    n_samples *= 2
    
for file in OUTPUT.values(): file.close()


