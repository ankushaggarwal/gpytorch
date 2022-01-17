###############################################################################
###################### Model and likelihood definitions #######################
###############################################################################

import torch
import sys
try:
    import gpytorch
except:
    from os.path import dirname, abspath
    sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
    import gpytorch
from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy2

###############################################################################
################################  Likelihood ##################################
###############################################################################

class CompositeLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self,likelihoods_list,indices):
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.indices = indices
        self.register_buffer('nu',torch.Tensor([1.]))
        #self.nu = torch.nn.parameter.Parameter(torch.Tensor([1.]),requires_grad=False) #1.
        self.register_buffer('small_slope',torch.Tensor([0.]))
        #self.small_slope = torch.nn.parameter.Parameter(torch.Tensor([0.]),requires_grad=False) #0.

    def set_indices(indices):
        self.indices = indices

    def forward(self,function_samples,**kwargs):
        #this does not work, but if expected_log_prob is implemented, this is not called for ELBO
        for i,l in enumerate(self.likelihoods_list):
            if i==0:
                p = l.forward(function_samples)
            else:
                p = torch.multiply(p,l.forward(function_samples))
        return p

    def split(self, full_y):
        y = []
        temp = torch.zeros_like(self.indices)
        #print(full_y)
        for i in range(self.indices.shape[-1]):
            temp[:,i] = self.indices[:,i]
            y.append(full_y[...,temp[self.indices]])
            temp[:,i] = False
        return y

    def expected_log_prob(self,observations, function_dist, *params,**kwargs):
        #split observations into a list of different parts observ[i]
        observ = self.split(observations)
        
        def log_prob_lambda(function_samples):
            #for the log_prob_lambda function, which takes function_samples as input
            #split the function_samples into a list of different parts f_samples[i]
            f_samples = self.split(function_samples)
            log_prob = []
            for i,l in enumerate(self.likelihoods_list):
                if isinstance(l,gpytorch.likelihoods.GaussianLikelihood):
                    log_prob.append(l(f_samples[i]).log_prob(observ[i]))
                elif isinstance(l,gpytorch.likelihoods.BernoulliLikelihood):
                    log_prob.append(gpytorch.functions.log_normal_cdf(f_samples[i].add(self.small_slope[0]).mul(observ[i].mul(2).sub(1)).mul(self.nu[0]))) #.mul(2).sub(1) changes the Bernoulli observations to -1 and 1, so that p(Y=y|f)=\Phi(yf), see the BernoulliLikelihood) for details on this
            
            #combine the log_prob back into the correct ordered full vector
            return torch.cat(log_prob,-1)

        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

#################################################################################
##################################### GP Model  #################################
#################################################################################

class GPModel(ApproximateGP):
    def __init__(self,n_inducing_points,ndim,lower_limits,upper_limits,deriv=2):
        #n_inducing_points = 200
        if deriv<1 or deriv>2:
            raise ValueError("deriv argument can only be either 1 or 2")
        inducing_points = torch.rand(n_inducing_points,ndim)
        for i in range(ndim):
            inducing_points[:,i] *= (upper_limits[i]-lower_limits[i])
            inducing_points[:,i] += lower_limits[i]
        inducing_index = torch.ones(n_inducing_points,deriv*ndim+1,dtype=bool)
        if deriv==2:
            inducing_index[:,ndim+1:]=False #do not use the 2nd derivatives for inducing points
        variational_distribution = CholeskyVariationalDistribution(torch.sum(inducing_index).item())
        variational_strategy = VariationalStrategy2(
            self, inducing_points, inducing_index, variational_distribution, learn_inducing_locations=True
        )
        super(GPModel, self).__init__(variational_strategy)
        if deriv==2:
            self.mean_module = gpytorch.means.LinearMeanGradGrad(ndim,bias=False)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGradGrad())
        elif deriv==1:
            self.mean_module = gpytorch.means.LinearMeanGrad(ndim,bias=False)
            self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())

        #self.mean_module = gpytorch.means.ConstantMeanGradGrad()
        #self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGradGrad())

    def forward(self, x, index):
        index = index.reshape(-1)
        mean_x = self.mean_module(x).reshape(-1)[index]
        full_kernel = self.covar_module(x)
        covar_x = full_kernel[..., index,:][...,:,index]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

