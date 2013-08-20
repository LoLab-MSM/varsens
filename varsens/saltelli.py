import ghalton
import numpy
import sys

class Varsens(object):
    '''The main variance sensitivity object which contains the core of the computation. It will
        execute the objective function n*(2*k+2) times.
        
        Parameters
        ----------
        objective : function
            a function that is passed k parameters resulting in a value or list of values
            to evaluate it's sensitivity
        scaling : function
            A function that when passed an array of numbers k long from [0..1] scales them to the
            desired range for the objective function. See varsens.scale for helpers.
        k : int
            Number of parameters that the objective function expects
        n : int
            Number of low discrepency draws to use to estimate the variance
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
            >>> v.var_y
            0.57105531939783061
            >>> v.E_2
            1.0040075966813939
            >>> v.sens
            array([ 0.58509195,  0.25311583,  0.03754574,  0.00752022,  0.00176028,
                    0.00177371])
            >>> v.sens_t
            array([  7.01277445e-01,   3.54584746e-01,   5.86321223e-02,
                     9.64126174e-03,   6.45996005e-04,   9.78965580e-04])
    '''
    def __init__(self, objective, scaling, k, n, verbose=True):
        self.k         = k
        self.n         = n
        self.objective = objective
        self.scaling   = scaling
        self.verbose   = verbose
        
        # Generate sample/re-sample space
        (M_1, M_2) = self.generate_sample()

        # Generate the sample/re-sample permutations
        N_j  = self.generate_N_j(M_1, M_2)                                # See Eq (11)
        N_nj = self.generate_N_j(M_2, M_1)

        # Execute the model to determine the objective function
        (fM_1, fM_2, fN_j, fN_nj) = self.objective_values(M_1, M_2, N_j, N_nj) 

        # From the model executions, compute the variable sensitivity
        self.compute_varsens(fM_1, fM_2, fN_j, fN_nj)

    def generate_sample(self):
        if self.verbose: print "Generating Low Discrepancy Sequence"
        
        seq = ghalton.Halton(self.k*2)
        seq.get(20*self.k) # Remove initial linear correlated points
        x = numpy.array(seq.get(self.n))
        M_1 = self.scaling(x[...,     0:self.k    ])
        M_2 = self.scaling(x[...,self.k:(2*self.k)])
        
        return M_1, M_2

    def move_spinner(self,i):
        '''A function to create a text spinner during long computations'''
        spin = ("|", "/","-", "\\")
        print " [%s] %d\r"%(spin[i%4],i),
        sys.stdout.flush()

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

    def objective_values(self, M_1, M_2, N_j, N_nj): #, fileobj=None):
        ''' Function parmeval calculates the fM_1, fM_2, and fN_j_i arrays needed for variance-based
        global sensitivity analysis as prescribed by Saltelli and derived from the work by Sobol
        (low-discrepancy sequences)
        '''

        # Determine objective return type
        test = self.objective(M_1[0])
        # assign the arrays that will hold fM_1, fM_2 and fN_j_n
        try:
            l = len(test)
            fM_1  = numpy.zeros([self.n] + [l])
            fM_2  = numpy.zeros([self.n] + [l])
            fN_j  = numpy.zeros([self.k] + [self.n] + [l])
            fN_nj = numpy.zeros([self.k] + [self.n] + [l])
        except TypeError:
            # assign the arrays that will hold fM_1, fM_2 and fN_j_n
            fM_1  = numpy.zeros(self.n)
            fM_2  = numpy.zeros(self.n)
            fN_j  = numpy.zeros([self.k] + [self.n]) # matrix is of shape (nparam, nsamples)
            fN_nj = numpy.zeros([self.k] + [self.n])

        # First process the A and B matrices
        if self.verbose: print "Processing f(M_1):"
        for i in range(self.n):
            fM_1[i]   = self.objective(M_1[i])
            if self.verbose: self.move_spinner(i)

        if self.verbose: print "Processing f(M_2):"
        for i in range(self.n):
            fM_2[i]   = self.objective(M_2[i])
            if self.verbose: self.move_spinner(i)

        if self.verbose: print "Processing f(N_j)"
        for i in range(self.k):
            if self.verbose: print " * parameter %d"%i
            for j in range(self.n):
                fN_j[i][j] = self.objective(N_j[i][j])
                if self.verbose: self.move_spinner(j)

        if self.verbose: print "Processing f(N_nj)"
        for i in range(self.k):
            if self.verbose: print " * parameter %d"%i
            for j in range(self.n):
                fN_nj[i][j] = self.objective(N_nj[i][j])
                if self.verbose: self.move_spinner(j)

        self.fM_1 = fM_1
        self.fM_2 = fM_2
        self.fN_j = fN_j
        self.fN_nj = fN_nj

        return fM_1, fM_2, fN_j, fN_nj

    def compute_varsens(self, fM_1, fM_2, fN_j, fN_nj):
        ''' Main computation of sensitivity via Saltelli method.
        '''
        if self.verbose: print "Final sensitivity calculation"

        self.E_2 = sum(fM_1*fM_2) / self.n      # Eq (21)

        #estimate V(y) from fM_1 and fM_2, paper uses only fM_1, this is a better estimate
        self.var_y = numpy.var(numpy.concatenate((fM_1, fM_2), axis=0), axis=0, ddof=1)

        # Estimate U_j and U_-j values and store them, but by double method
        self.U_j  =  numpy.sum(fM_1 * fN_j,  axis=1) / (self.n - 1)  # Eq (12)
        self.U_j  += numpy.sum(fM_2 * fN_nj, axis=1) / (self.n - 1) 
        self.U_j  /= 2.0
        self.U_nj =  numpy.sum(fM_1 * fN_nj, axis=1) / (self.n - 1)  # Eq (unnumbered one after 18)
        self.U_nj += numpy.sum(fM_2 * fN_j,  axis=1) / (self.n - 1) 
        self.U_nj /= 2.0
        
        #allocate the S_i and ST_i arrays
        if len(self.U_j.shape) == 1:
            self.sens   = numpy.zeros(self.k)
            self.sens_t = numpy.zeros(self.k)
        else:
            self.sens   = numpy.zeros([self.k]+[self.U_j.shape[1]])
            self.sens_t = numpy.zeros([self.k]+[self.U_j.shape[1]])

        # now get the S_i and ST_i, Eq (27) & Eq (28)
        for j in range(self.k):
            self.sens[j]   = (self.U_j[j] - self.E_2) / self.var_y
            self.sens_t[j] = 1.0 - ((self.U_nj[j]- self.E_2) / self.var_y)

        # Compute 2nd order terms (from double estimates)
        self.sens_2  =  numpy.tensordot(fN_nj, fN_j,  axes=([1],[1]))
        self.sens_2  += numpy.tensordot(fN_j,  fN_nj, axes=([1],[1]))
        self.sens_2  /= 2.0*(self.n-1)
        self.sens_2  -= self.E_2
        self.sens_2  /= self.var_y
        
        self.sens_2n =  numpy.tensordot(fN_nj, fN_nj, axes=([1],[1]))
        self.sens_2n += numpy.tensordot(fN_j,  fN_j,  axes=([1],[1]))
        self.sens_2n /= 2.0 * (self.n-1)
        self.sens_2n -= self.E_2
        self.sens_2n /= self.var_y

