import ghalton
import numpy
import sys

class Varsens(object):
    """The main variance sensitivity object which contains the core of the computation"""
    
    def __init__(self, objective, scaling, k, n, verbose=True):
        self.k = k
        self.n = n
        self.objective = objective
        self.scaling = scaling
        self.verbose = verbose

        if verbose: print "Generating Low Discrepancy Sequence"

        seq = ghalton.Halton(k*2)
        seq.get(20*k) # Remove initial linear correlated points
        x = numpy.array(seq.get(n))
        M_1 = scaling(x[...,0:k    ])
        M_2 = scaling(x[...,k:(2*k)])
    
        N_j  = self.generate_N_j(M_1, M_2)                                # See Eq (11)
        N_nj = self.generate_N_j(M_2, M_1)
    
        (fM_1, fM_2, fN_j, fN_nj) = self.objective_values(M_1, M_2, N_j, N_nj) 
    
        if verbose: print "Final sensitivity calculation"
        self.compute_varsens(fM_1, fM_2, fN_j, fN_nj)

    def move_spinner(self,i):
        """A function to create a text spinner during long computations"""
        spin = ("|", "/","-", "\\")
        print "[%s] %d\r"%(spin[i%4],i),
        sys.stdout.flush()

    def generate_N_j(self, M_1, M_2):
        """when passing the quasi-random low discrepancy-treated A and B matrixes, this function
        iterates over all the possibilities and returns the C matrix for simulations.
        See e.g. Saltelli, Ratto, Andres, Campolongo, Cariboni, Gatelli, Saisana,
        Tarantola Global Sensitivity Analysis"""

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

        # assign the arrays that will hold fM_1, fM_2 and fN_j_n
        fM_1  = numpy.zeros(self.n)
        fM_2  = numpy.zeros(self.n)
        fN_j  = numpy.zeros([self.k] + [self.n]) # matrix is of shape (nparam, nsamples)
        fN_nj = numpy.zeros([self.k] + [self.n])

        # First process the A and B matrices
        if self.verbose: print "Processing f(M_1):"
        for i in range(M_1.shape[0]):
            fM_1[i]   = self.objective(M_1[i])
            if self.verbose: self.move_spinner(i)

        if self.verbose: print "Processing f(M_2):"
        for i in range(M_2.shape[0]):
            fM_2[i]   = self.objective(M_2[i])
            if self.verbose: self.move_spinner(i)

        if self.verbose: print "Processing f(N_j)"
        for i in range(N_j.shape[0]):
            if self.verbose: print " * parameter %d"%i
            for j in range(N_j.shape[1]):
                fN_j[i][j] = self.objective(N_j[i][j])
                if self.verbose: self.move_spinner(j)

        if self.verbose: print "Processing f(N_nj)"
        for i in range(N_j.shape[0]):
            if self.verbose: print " * parameter %d"%i
            for j in range(N_j.shape[1]):
                fN_nj[i][j] = self.objective(N_nj[i][j])
                if self.verbose: self.move_spinner(j)

        return fM_1, fM_2, fN_j, fN_nj

    def compute_varsens(self, fM_1, fM_2, fN_j, fN_nj):
        self.k   = fN_j.shape[0] # should be the number of parameters
        self.n = fN_j.shape[1] # should be the number of samples from the original matrix

        self.E_2 = sum(fM_1*fM_2) / self.n      # Eq (21)

        # Estimate U_j and U_-j values and store them 
        U_j  = numpy.sum(fM_1 * fN_j,  axis=1) / (self.n - 1)  # Eq (12)
        U_nj = numpy.sum(fM_1 * fN_nj, axis=1) / (self.n - 1)  # Eq (unnumbered one after 18)

        #estimate V(y) from fM_1 and fM_2, paper uses only fM_1, this is a better estimate
        self.var_y = (numpy.var(fM_1, axis=0, ddof=1)+numpy.var(fM_2, axis=0, ddof=1))/2.0

        #allocate the S_i and ST_i arrays
        self.sens   = numpy.zeros(self.k)
        self.sens_t = numpy.zeros(self.k)

        # now get the S_i and ST_i, Eq (27) & Eq (28)
        for j in range(self.k):
            self.sens[j]   =       ((U_j[j] - self.E_2) / self.var_y)
            self.sens_t[j] = 1.0 - ((U_nj[j]- self.E_2) / self.var_y)

