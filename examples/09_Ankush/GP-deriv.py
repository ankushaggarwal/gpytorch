import torch
import sys
from os.path import dirname, abspath
sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
import gpytorch
import math
from matplotlib import pyplot as plt
import numpy as np

lb, ub = 0.0, 5*math.pi
n1 = 1 #function values
train_x1 = torch.linspace(lb, ub, n1)#.unsqueeze(-1)
train_y1 = torch.sin(2*train_x1) + torch.cos(train_x1) + 0.05 * torch.randn(train_x1.size())

n2=50 #derivative values at different x locations
train_x2 = torch.linspace(lb, ub, n2)#.unsqueeze(-1)
train_y2 = -torch.sin(train_x2) + 2*torch.cos(2*train_x2) + 0.05 * torch.randn(train_x2.size())

train_x = torch.cat([train_x1 , train_x2])
train_y = torch.cat([train_y1,train_y2])

ndata,ndim = train_x.shape.numel(),1
train_index = torch.empty(ndata,ndim+1,dtype=bool)
train_index[:n1,0]=True
train_index[:n1,1]=False
train_index[n1:,0]=False
train_index[n1:,1]=True

#train_index = torch.tensor([True]*n1+[False]*n2)

class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.base_kernel = gpytorch.kernels.RBFKernelGrad()
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x, index):
        index = index.reshape(-1)
        mean_x = self.mean_module(x).reshape(-1)[index]
        full_kernel = self.covar_module(x)
        covar_x = full_kernel[..., index,:][...,:,index]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = GPModelWithDerivatives((train_x,train_index), train_y, likelihood)

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

#print(train_y)
#output = model(train_x1,train_x2)
#loss = -mll(output, train_y)
#loss.backward()

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x,train_index)
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()


# Set into eval mode
model.eval()
likelihood.eval()

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 6))

n11=200
test_x = torch.linspace(lb, ub, n11)
test_index = torch.ones(test_x.shape[0],ndim+1,dtype=bool)
#test_index = torch.tensor([True]*n11+[False]*n22)

# Make predictions
with torch.no_grad(), gpytorch.settings.max_cg_iterations(50):
    predictions = likelihood(model(test_x,test_index))
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
y2_ax.plot(train_x[n1:].detach().numpy(), train_y[n1:].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x.detach().numpy(), mean[1::2].detach().numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(test_x.detach().numpy(), lower[1::2].detach().numpy(), upper[1::2].detach().numpy(), alpha=0.5)
y2_ax.legend(['Observed Derivatives', 'Mean', 'Confidence'])
y2_ax.set_title('Derivatives')

plt.show()
