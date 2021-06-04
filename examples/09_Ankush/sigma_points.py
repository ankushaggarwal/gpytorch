import numpy as np

class MerweScaledSigmaPoints2(object):
 
    def __init__(self, n, alpha=1e-3, beta=2., kappa=0., sqrt_method=None, subtract=None):
        """ Generates sigma points and weights according to Van der Merwe's
        2004 dissertation[1]. It parametizes the sigma points using
        alpha, beta, kappa terms, and is the version seen in most publications.
 
        Unless you know better, this should be your default choice.
 
        Parameters
        ----------
 
        n : int
            Dimensionality of the state. 2n+1 weights will be generated.
 
        alpha : float
            Determins the spread of the sigma points around the mean.
            Usually a small positive value (1e-3) according to [3].
 
        beta : float
            Incorporates prior knowledge of the distribution of the mean. For
            Gaussian x beta=2 is optimal, according to [3].
 
        kappa : float, default=0.0
            Secondary scaling parameter usually set to 0 according to [4],
            or to 3-n according to [5].
 
        sqrt_method : function(ndarray), default=scipy.linalg.cholesky
            Defines how we compute the square root of a matrix, which has
            no unique answer. Cholesky is the default choice due to its
            speed. Typically your alternative choice will be
            scipy.linalg.sqrtm. Different choices affect how the sigma points
            are arranged relative to the eigenvectors of the covariance matrix.
            Usually this will not matter to you; if so the default cholesky()
            yields maximal performance. As of van der Merwe's dissertation of
            2004 [6] this was not a well reseached area so I have no advice
            to give you.
 
            If your method returns a triangular matrix it must be upper
            triangular. Do not use numpy.linalg.cholesky - for historical
            reasons it returns a lower triangular matrix. The SciPy version
            does the right thing.
 
        subtract : callable (x, y), optional
            Function that computes the difference between x and y.
            You will have to supply this if your state variable cannot support
            subtraction, such as angles (359-1 degreees is 2, not 358). x and y
            are state vectors, not scalars.
 
        References
        ----------
 
        .. [1] R. Van der Merwe "Sigma-Point Kalman Filters for Probabilitic
               Inference in Dynamic State-Space Models" (Doctoral dissertation)
 
        """
 
        self.n = n
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa
        if sqrt_method is None:
            self.sqrt = cholesky
        else:
            self.sqrt = sqrt_method
 
        if subtract is None:
            self.subtract= np.subtract
        else:
            self.subtract = subtract
 
 
    def num_sigmas(self):
        """ Number of sigma points for each variable in the state x"""
        return 2*self.n + 1
 
 
    def sigma_points(self, x, P):
        """ Computes the sigma points for an unscented Kalman filter
        given the mean (x) and covariance(P) of the filter.
        Returns tuple of the sigma points and weights.
 
        Works with both scalar and array inputs:
        sigma_points (5, 9, 2) # mean 5, covariance 9
        sigma_points ([5, 2], 9*eye(2), 2) # means 5 and 2, covariance 9I
 
        Parameters
        ----------
 
        X An array-like object of the means of length n
            Can be a scalar if 1D.
            examples: 1, [1,2], np.array([1,2])
 
        P : scalar, or np.array
           Covariance of the filter. If scalar, is treated as eye(n)*P.
 
        Returns
        -------
 
        sigmas : np.array, of size (n, 2n+1)
            Two dimensional array of sigma points. Each column contains all of
            the sigmas for one dimension in the problem space.
 
            Ordered by Xi_0, Xi_{1..n}, Xi_{n+1..2n}
        """
 
        n = self.n
 
        if  np.isscalar(P):
            P = np.eye(n)*P
        else:
            P = np.asarray(P)
 
        lambda_ = self.alpha**2 * (n + self.kappa) - n
        U = self.sqrt((lambda_ + n)*P)
 
        sigmas = np.zeros((n, 2*n+1))
        sigmas[:,0] = x
        for k in range(n):
            sigmas[:, k+1]   = self.subtract(x, -U[k])
            sigmas[:, n+k+1] = self.subtract(x, U[k])
 
        return sigmas
 
 
    def weights(self):
        """ Computes the weights for the scaled unscented Kalman filter.
 
        Returns
        -------
 
        Wm : ndarray[2n+1]
            weights for mean
 
        Wc : ndarray[2n+1]
            weights for the covariances
        """
 
        n = self.n
        lambda_ = self.alpha**2 * (n +self.kappa) - n
 
        c = .5 / (n + lambda_)
        Wc = np.full(2*n + 1, c)
        Wm = np.full(2*n + 1, c)
        Wc[0] = lambda_ / (n + lambda_) + (1 - self.alpha**2 + self.beta)
        Wm[0] = lambda_ / (n + lambda_)
 
        return Wm, Wc

x=np.zeros(2)
P=np.diag([0.1,0.2])
sp = MerweScaledSigmaPoints2(len(x),alpha=1.,sqrt_method=np.sqrt)
print(sp.sigma_points(x,P))
print(sp.weights())
