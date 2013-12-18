import ghalton
import numpy
import sys
import random

def move_spinner(i):
    '''A function to create a text spinner during long computations'''
    spin = ("|", "/","-", "\\")
    print " [%s] %d\r" % (spin[i%4],i)
    sys.stdout.flush()

class Sample(object):
    ''' An object containing the definition of the sample space, as well as the 
        actual matrices M_1 and M_2. Generated via Halton low-discrepency
        sequence
    '''
    def __init__(self, k, n, scaling, verbose = True, loadFile = None):
        self.k = k
        self.n = n
        self.scaling = scaling
        self.verbose = verbose

        if scaling == None: return None

        if self.verbose: print "Generating Low Discrepancy Sequence"
        
        if loadFile == None:
            seq = ghalton.Halton(self.k*2)
            seq.get(20*self.k) # Remove initial linear correlated points
            x = numpy.array(seq.get(self.n))
            self.M_1 = self.scaling(x[...,     0:self.k    ])
            self.M_2 = self.scaling(x[...,self.k:(2*self.k)])
            
            # This little shuffle enormously improves the performance
            numpy.random.shuffle(self.M_2) # Eliminate any correlation
        else:
            x = numpy.loadtxt(open(loadFile, "rb"), delimiter=",")
            self.M_1 = self.scaling(x[     0:self.n,    ...])
            self.M_2 = self.scaling(x[self.n:(2*self.n),...])
            random.seed(1)
            random.shuffle(self.M_2)

        # Generate the sample/re-sample permutations
        self.N_j  = self.generate_N_j(self.M_1, self.M_2) # See Eq (11)
        self.N_nj = self.generate_N_j(self.M_2, self.M_1)

    def generate_N_j(self, M_1, M_2):
        '''when passing the quasi-random low discrepancy-treated A and B matrixes, this function
        iterates over all the possibilities and returns the C matrix for simulations.
        See e.g. Saltelli, Ratto, Andres, Campolongo, Cariboni, Gatelli, Saisana,
        Tarantola Global Sensitivity Analysis'''

        # allocate the space for the C matrix
        N_j = numpy.array([M_2]*self.k)

        # Now we have nparams copies of M_2. replace the i_th column of N_j with the i_th column of M_1
        for i in range(self.k):
            N_j[i,:,i] = M_1[:,i]
        return N_j

    def flat(self):
        '''Return the sample space as an array n*(2*k+2) long, containing arrays k long'''
        x = numpy.append(self.M_1, self.M_2, axis=0)
        for i in range(self.k):
            x = numpy.append(x, self.N_j[i], axis=0)
        for i in range(self.k):
            x = numpy.append(x, self.N_nj[i], axis=0)
        return x

    def export(self, prefix, postfix, blocksize):
        f = self.flat()
        for b in range(int(numpy.ceil(1.0*len(f) / blocksize))):
            numpy.savetxt("%s%d%s" % (prefix, b+1, postfix), f[b*blocksize : (b+1)*blocksize])

class Objective(object):
    ''' Function parmeval calculates the fM_1, fM_2, and fN_j_i arrays needed for variance-based
    global sensitivity analysis as prescribed by Saltelli and derived from the work by Sobol
    (low-discrepancy sequences)
    '''
    def __init__(self, k, n, sample, objective_func, verbose=True):
        self.k              = k
        self.n              = n
        self.sample         = sample
        self.objective_func = objective_func
        self.verbose        = verbose
        self.fM_1           = None
        self.fM_2           = None
        self.fN_j           = None
        self.fN_nj          = None

        if objective_func == None: return # If no function specified, then user will fill

        # Determine objective_func return type
        test = self.objective_func(sample.M_1[0])

        # assign the arrays that will hold fM_1, fM_2 and fN_j_n, either as a list or single value
        try:
            l = len(test)
            self.fM_1  = numpy.zeros([self.n] + [l])
            self.fM_2  = numpy.zeros([self.n] + [l])
            self.fN_j  = numpy.zeros([self.k] + [self.n] + [l])
            self.fN_nj = numpy.zeros([self.k] + [self.n] + [l])
        except TypeError:
            # assign the arrays that will hold fM_1, fM_2 and fN_j_n
            self.fM_1  = numpy.zeros(self.n)
            self.fM_2  = numpy.zeros(self.n)
            self.fN_j  = numpy.zeros([self.k] + [self.n]) # matrix is of shape (nparam, nsamples)
            self.fN_nj = numpy.zeros([self.k] + [self.n])

        # First process the A and B matrices
        if self.verbose: print "Processing f(M_1):"
        self.fM_1[0] = test # Save first execution
        for i in range(1,self.n):
            self.fM_1[i]   = self.objective_func(sample.M_1[i])
            if self.verbose: move_spinner(i)

        if self.verbose: print "Processing f(M_2):"
        for i in range(self.n):
            self.fM_2[i]   = self.objective_func(sample.M_2[i])
            if self.verbose: move_spinner(i)

        if self.verbose: print "Processing f(N_j)"
        for i in range(self.k):
            if self.verbose: print " * parameter %d"%i
            for j in range(self.n):
                self.fN_j[i][j] = self.objective_func(sample.N_j[i][j])
                if self.verbose: move_spinner(j)

        if self.verbose: print "Processing f(N_nj)"
        for i in range(self.k):
            if self.verbose: print " * parameter %d"%i
            for j in range(self.n):
                self.fN_nj[i][j] = self.objective_func(sample.N_nj[i][j])
                if self.verbose: move_spinner(j)

    def load(self, prefix, postfix, count, scaling = 1.0):
        d = numpy.loadtxt("%s%d%s" % (prefix, 1, postfix))
        for c in range(1, count):
            d = numpy.append(d, numpy.loadtxt("%s%d%s" % (prefix, c+1, postfix)), axis=0)

        # assign the arrays that will hold fM_1, fM_2 and fN_j_n, either as a list or single value
        if(len(d.shape) > 1):
            l = len(d[0])
            self.fM_1  = numpy.zeros([self.n] + [l])
            self.fM_2  = numpy.zeros([self.n] + [l])
            self.fN_j  = numpy.zeros([self.k] + [self.n] + [l])
            self.fN_nj = numpy.zeros([self.k] + [self.n] + [l])
        else:
            # assign the arrays that will hold fM_1, fM_2 and fN_j_n
            self.fM_1  = numpy.zeros(self.n)
            self.fM_2  = numpy.zeros(self.n)
            self.fN_j  = numpy.zeros([self.k] + [self.n]) # matrix is of shape (nparam, nsamples)
            self.fN_nj = numpy.zeros([self.k] + [self.n])
        
        # Put into appropriate bins
        self.fM_1 = d[0   : self.n] / scaling
        self.fM_2 = d[self.n : 2*self.n]  / scaling
        for i in range(self.k):
            self.fN_j[i] = d[(2+i)*self.n : (3+i)*self.n ] / scaling
        for i in range(self.k):
            self.fN_nj[i] = d[(2+i+self.k)*self.n : (3+i+self.k)*self.n ] / scaling
        ######################
        # Trim the row from *all* matricies if *one* has a nan
        nans = [] 
        # locate nans
        isnan = numpy.logical_or(numpy.isnan(self.fM_1), numpy.isnan(self.fM_2))
        for i in range(self.k):
            isnan = numpy.logical_or(isnan, numpy.isnan(self.fN_j[i]))
            isnan = numpy.logical_or(isnan, numpy.isnan(self.fN_nj[i]))
        # If there are multiple objectives, use the first column
        if len(isnan.shape) > 1: isnan = isnan[:,0]

        # Construct array of NaN rows
        for i in range(len(isnan)):
            if isnan[i]: nans.append(i)
        
        # Now delete the located nans
        self.fM_1     = numpy.delete(self.fM_1,  nans, axis=0)
        self.fM_2     = numpy.delete(self.fM_2,  nans, axis=0)
        self.fN_j     = numpy.delete(self.fN_j,  nans, axis=1)
        self.fN_nj    = numpy.delete(self.fN_nj, nans, axis=1)

        if(len(nans) > 0):
            print "WARNING: %d of %d objectives were NaN, %lf%% loss\r" % (len(nans),self.n, 100.0*len(nans)/self.n)

class Varsens(object):
    '''The main variance sensitivity object which contains the core of the computation. It will
        execute the objective function n*(2*k+2) times.
        
        Parameters
        ----------
        objective : function or Objective
            a function that is passed k parameters resulting in a value or 
            list of values to evaluate it's sensitivity, or a pre-evaluated set
            of objectives in an Objective object
        scaling_func : function
            A function that when passed an array of numbers k long from [0..1]
            scales them to the desired range for the objective function. See
            varsens.scale for helpers.
        k : int
            Number of parameters that the objective function expects
        n : int
            Number of low discrepency draws to use to estimate the variance
        sample : Sample
            A predefined sample. If specified, the k, n and scaling_func variables
            are ignored, and this is used as the sample
        verbose : bool
            Whether or not to print progress in computation

        Returns
        -------
        Varsens object. It will contain var_y, E_2, sens and sens_t.

        Examples
        ________
            >>> from varsens    import *
            >>> import numpy
            >>> def gi_function(xi, ai): return (numpy.abs(4.0*xi-2.0)+ai) / (1.0+ai)
            ... 
            >>> def g_function(x, a): return numpy.prod([gi_function(xi, a[i]) for i,xi in enumerate(x)])
            ... 
            >>> def g_scaling(x): return x
            ... 
            >>> def g_objective(x): return g_function(x, [0, 0.5, 3, 9, 99, 99])
            ... 
            >>> v = Varsens(g_objective, g_scaling, 6, 1024, verbose=False)
            >>> v.var_y # doctest: +ELLIPSIS
            0.5...
            >>> v.sens # doctest: +ELLIPSIS
            array([...])
            >>> v.sens_t # doctest: +ELLIPSIS
            array([...])
    '''
    def __init__(self, objective, scaling_func=None, k=None, n=None, sample=None, verbose=True):
        self.verbose    = verbose

        # If the sample object if predefined use it
        if isinstance(sample, Sample):
            self.sample = sample
            self.k      = sample.k
            self.n      = sample.n
        elif k != None and n != None and scaling_func != None: # Create sample from space definition
            self.k      = k
            self.n      = n
            self.sample = Sample(k, n, scaling_func, verbose)
        elif not isinstance(objective, Objective):
            self.k      = objective.k
            self.n      = objective.n
            # No sample provided, no sample space definition provided, no pre-evaluated objective provided
            # Impossible to compute variable sensitivity
            raise ValueError("Must specify sample, (k,n,scaling_func), or Objective object")

        # Execute the model to determine the objective function
        if isinstance(objective, Objective):
            self.objective = objective
        else: # The object is predefined.
            self.objective = Objective(self.k, self.n, self.sample, objective, verbose)

        # From the model executions, compute the variable sensitivity
        self.compute_varsens()

    def compute_varsens(self):
        ''' Main computation of sensitivity via Saltelli method.
        '''
        if self.verbose: print "Final sensitivity calculation"

        n = len(self.objective.fM_1)
        self.E_2 = sum(self.objective.fM_1*self.objective.fM_2) / n      # Eq (21)
        #self.E_2 = sum(self.objective.fM_1) / n # Eq(22)
        #self.E_2 *= self.E_2
        
        #estimate V(y) from self.objective.fM_1 and self.objective.fM_2
        # paper uses only self.objective.fM_1, this is a better estimator
        self.var_y = numpy.var(numpy.concatenate((self.objective.fM_1, self.objective.fM_2), axis=0), axis=0, ddof=1)

# FIXME: This NEED WORK, and it is IMPORTANT
        #if not numpy.all(numpy.sqrt(numpy.abs(self.E_2)) > 1.96*numpy.sqrt(self.var_y / n)):
        #    print "Excessive variance in estimation of E^2"
        #    raise ArithmeticError

        # Estimate U_j and U_-j values and store them, but by double method
        self.U_j  =  numpy.sum(self.objective.fM_1 * self.objective.fN_j,  axis=1) / (n - 1)  # Eq (12)
       # self.U_j  += numpy.sum(self.objective.fM_2 * self.objective.fN_nj, axis=1) / (n - 1) 
        #self.U_j  /= 2.0                                                              
        self.U_nj =  numpy.sum(self.objective.fM_1 * self.objective.fN_nj, axis=1) / (n - 1)  # Eq (unnumbered one after 18)
        #self.U_nj += numpy.sum(self.objective.fM_2 * self.objective.fN_j,  axis=1) / (n - 1) 
        #self.U_nj /= 2.0
        
        #allocate the S_i and ST_i arrays
        if len(self.U_j.shape) == 1:
            self.sens   = numpy.zeros(self.k)
            self.sens_t = numpy.zeros(self.k)
        else: # It's a list of values
            self.sens   = numpy.zeros([self.k]+[self.U_j.shape[1]])
            self.sens_t = numpy.zeros([self.k]+[self.U_j.shape[1]])

        # now get the S_i and ST_i, Eq (27) & Eq (28)
        for j in range(self.k):
            self.sens[j]   = (self.U_j[j] ) / self.var_y
            self.sens_t[j] = 1.0 - ((self.U_nj[j]) / self.var_y)

        # Compute 2nd order terms (from double estimates)
        self.sens_2  =  numpy.tensordot(self.objective.fN_nj, self.objective.fN_j,  axes=([1],[1]))
        self.sens_2  += numpy.tensordot(self.objective.fN_j,  self.objective.fN_nj, axes=([1],[1]))
        self.sens_2  /= 2.0*(n-1)
        self.sens_2  -= self.E_2
        self.sens_2  /= self.var_y
        
        self.sens_2n =  numpy.tensordot(self.objective.fN_nj, self.objective.fN_nj, axes=([1],[1]))
        self.sens_2n += numpy.tensordot(self.objective.fN_j,  self.objective.fN_j,  axes=([1],[1]))
        self.sens_2n /= 2.0 * (n-1)
        self.sens_2n -= self.E_2
        self.sens_2n /= self.var_y

        # Numerical error can make some values exceed what is sensible
        self.sens    = numpy.clip(self.sens,    0, 1)
        self.sens_t  = numpy.clip(self.sens_t,  0, 1e6)
        self.sens_2  = numpy.clip(self.sens_2,  0, 1)
        self.sens_2n = numpy.clip(self.sens_2n, 0, 1)
