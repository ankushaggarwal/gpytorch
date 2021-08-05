import torch
import sys
from os.path import dirname, abspath
sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
import gpytorch
import math

class CompositeLikelihood(gpytorch.likelihoods._OneDimensionalLikelihood):
    def __init__(self,likelihoods_list,indices):
        super().__init__()
        self.likelihoods_list = likelihoods_list
        self.indices = indices
        self.nu = 1.

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
                    log_prob.append(gpytorch.functions.log_normal_cdf(f_samples[i].mul(observ[i].mul(2).sub(1)).mul(self.nu))) #.mul(2).sub(1) changes the Bernoulli observations to -1 and 1, so that p(Y=y|f)=\Phi(yf), see the BernoulliLikelihood) for details on this
            
            #combine the log_prob back into the correct ordered full vector
            return torch.cat(log_prob,-1)

        log_prob = self.quadrature(log_prob_lambda, function_dist)
        return log_prob

lb, ub = 0.0, 1. #5*math.pi
n1 = 40 #function values
freq = 2 #frequency of the size function
train_x1 = torch.linspace(lb, ub, n1)#.unsqueeze(-1)
train_y1 = torch.sin(freq*train_x1) + 0.005 * torch.randn(train_x1.size())

n2=50 #derivative values at different x locations
train_x2 = torch.linspace(lb, ub, n2)#.unsqueeze(-1)
train_y2 = torch.ones_like(train_x2) #freq*torch.cos(freq*train_x2) + 0.005 * torch.randn(train_x2.size())

train_x = torch.cat([train_x1 , train_x2])
train_y = torch.cat([train_y1,train_y2])

ndata,ndim = train_x.shape.numel(),1
train_index = torch.empty(ndata,ndim+1,dtype=bool)
train_index[:n1,0]=True
train_index[:n1,1]=False
train_index[n1:,0]=False
train_index[n1:,1]=True

from gpytorch.models import ApproximateGP
from gpytorch.variational import CholeskyVariationalDistribution
from gpytorch.variational import VariationalStrategy2

class GPModel(ApproximateGP):
    def __init__(self):
        inducing_points = torch.rand(40)*ub
        inducing_index = torch.ones(40,ndim+1,dtype=bool)
        inducing_index[:,1]=False
        variational_distribution = CholeskyVariationalDistribution(torch.sum(inducing_index).item())
        variational_strategy = VariationalStrategy2(
            self, inducing_points, inducing_index, variational_distribution, learn_inducing_locations=True
        )
        super(GPModel, self).__init__(variational_strategy)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())

    def forward(self, x, index):
        index = index.reshape(-1)
        mean_x = self.mean_module(x).reshape(-1)[index]
        full_kernel = self.covar_module(x)
        covar_x = full_kernel[..., index,:][...,:,index]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

model = GPModel()
likelihood  = gpytorch.likelihoods.GaussianLikelihood()
likelihood2 = gpytorch.likelihoods.BernoulliLikelihood()

test_likelihood = CompositeLikelihood((likelihood,likelihood2),train_index)

#pp = (model(test_x,x_index=test_index))
#test(pp)

training_iter = 200 

# Find optimal model hyperparameters
model.train()
test_likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    {'params': likelihood.parameters()},
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
#mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
mll = gpytorch.mlls.VariationalELBO(test_likelihood, model, num_data=train_y.size(0))

#print(train_y)
#output = model(train_x1,train_x2)
#loss = -mll(output, train_y)
#loss.backward()

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x,x_index=train_index)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()

#increase the nu and train again
test_likelihood.nu = 10.
for i in range(training_iter//3):
    optimizer.zero_grad()
    output = model(train_x,x_index=train_index)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iter, loss.item()))
    optimizer.step()
model.eval()

from matplotlib import pyplot as plt
import numpy as np
# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 6))

n11=200
test_x = torch.linspace(lb, ub, n11)
test_index = torch.ones(test_x.shape[0],ndim+1,dtype=bool)
#test_index = torch.tensor([True]*n11+[False]*n22)

# Make predictions
with torch.no_grad(), gpytorch.settings.max_cg_iterations(50):
    predictions = model(test_x,x_index=test_index)
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# Plot training data as black stars
y1_ax.plot(train_x[:n1].detach().numpy(), train_y[:n1].detach().numpy(), 'k*')
# Predictive mean as blue line
y1_ax.plot(test_x.detach().numpy(), mean[::2].detach().numpy(), 'b')
# Shade in confidence
y1_ax.fill_between(test_x.detach().numpy(), lower[::2].detach().numpy(), upper[::2].detach().numpy(), alpha=0.5)
y1_ax.legend(['Observed Values', 'Mean', 'Confidence'])
y1_ax.set_title('Function values')

# Plot training data as black stars
#y2_ax.plot(train_x[n1:].detach().numpy(), train_y[n1:].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x.detach().numpy(), mean[1::2].detach().numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(test_x.detach().numpy(), lower[1::2].detach().numpy(), upper[1::2].detach().numpy(), alpha=0.5)
y2_ax.legend(['Observed Derivatives', 'Mean', 'Confidence'])
y2_ax.set_title('Derivatives')

plt.show()
