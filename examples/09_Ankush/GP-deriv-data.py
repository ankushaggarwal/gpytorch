import numpy as np

data = np.load('MV18242PL-FS.npz')
F = data['F']
S = data['S']

def invariants(F):
        M = np.array([1,0,0])
        M = M/np.linalg.norm(M)
        Fall = np.reshape(F,(-1,3,3))
        I1,I4 = [], []
        for f in Fall:
                C = np.dot(f.T,f)
                I1.append(np.trace(C))
                I4.append(np.dot(M,np.dot(C,M)))
        return np.reshape(np.array(I1),np.shape(F)[:-2]), np.reshape(np.array(I4),np.shape(F)[:-2])

def principal_stretches(F):
        Fall = np.reshape(F,(-1,3,3))
        l1,l2 = [], []
        for f in Fall:
                l1.append(f[0,0]) #assumes a biaxial stretching
                l2.append(f[1,1])
        return np.reshape(np.array(l1),np.shape(F)[:-2]), np.reshape(np.array(l2),np.shape(F)[:-2])

def stresses(S):
        Sall = np.reshape(S,(-1,3,3))
        s11, s22 = [], []
        for s in Sall:
                s11.append(s[0,0])
                s22.append(s[1,1]) #assumes the shear stresses are zero
        return np.reshape(np.array(s11),np.shape(S)[:-2]), np.reshape(np.array(s22),np.shape(S)[:-2])


I1, I4 = invariants(F)
lam1, lam2 = principal_stretches(F)
s11, s22 = S[:,:,0,0], S[:,:,1,1]

print(np.shape(I1)[0], 'protocols read')

def partial_derivs(S11,S22,Lam1,Lam2):
        d1,d2 = [], []
        A = np.zeros([2,2])
        r = np.zeros(2)
        for (s11,s22,lam1,lam2) in zip(S11.flatten(),S22.flatten(),Lam1.flatten(),Lam2.flatten()):
                A[0,0] = 2*(lam1**2-1./lam1**2/lam2**2)
                A[1,0] = 2*(lam2**2-1./lam1**2/lam2**2)
                A[0,1] = 2*lam1**2 #assumes the fiber direction is the first axis
                r[0],r[1] = s11,s22
                x = np.linalg.solve(A,r)
                d1.append(x[0])
                d2.append(x[1])
        return np.reshape(np.array(d1),np.shape(S11)), np.reshape(np.array(d2),np.shape(S11))

dWdI1, dWdI4 = partial_derivs(s11,s22,lam1,lam2)

########################################################################################
########################################   GPYTORCH PART ###############################
########################################################################################
#GP code
#Import the GPyTorch libraries
import math
import torch
import sys
from os.path import dirname, abspath
sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
import gpytorch
import gpytorch
from matplotlib import pyplot as plt

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernelGrad())

    def forward(self, x, index):
        index = index.squeeze()
        index = torch.stack((index,~index)).T.reshape(-1)
        mean_x = self.mean_module(x)
        full_kernel = self.covar_module(x)
        covar_x = full_kernel[...,index,:][...,index]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#Convert the numpy arrays to torch tensors
x=I1[0][dWdI1[0]>0]
y=dWdI1[0][dWdI1[0]>0]
train_x1 = torch.tensor([3.]) 
train_y1 = torch.tensor([0.]) #the value of W at I1=3
n1 = 1
train_x2 = torch.tensor(x.flatten())
train_y2 = torch.tensor(y.flatten()*10) #the derivative of W from experiment
n2 = len(x)
train_x = torch.cat([train_x1 , train_x2])
train_y = torch.cat([train_y1,train_y2])

train_index = torch.tensor([True]*n1+[False]*n2)
# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel((train_x,train_index),train_y,likelihood)
#model.covar_module.base_kernel.lengthscale = 0.5
#model.likelihood.noise = 0.05

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.9)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

print('Starting the training')
training_iter = 50
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x,train_index)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
'''
def predict_model(test_x1):
    test_x1 = np.insert(test_x1,0,1.)
    # Make predictions by feeding model through likelihood
    with gpytorch.settings.fast_pred_var():
        test_x = torch.tensor(test_x1,dtype=torch.float64,requires_grad=True)
        observed_pred = (model(test_x))
        dydtest_x = torch.autograd.grad(observed_pred.mean.sum(), test_x)[0]
        #dydtest_x = torch.tensor([torch.autograd.grad(observed_pred.mean[i], test_x, retain_graph=True)[0][i] for i, _ in enumerate(test_x)])
    residual_stress = observed_pred.mean.detach().numpy()[0]
    return (observed_pred.mean.detach().numpy()[1:]-residual_stress, dydtest_x.numpy()[1:])

#the above gives us the stresses and derivatives of the stresses at requested points. An example is:
#print(predict_model(np.linspace(min(x_data2), max(x_data2),30)))
'''

# Initialize plots
f, (y1_ax, y2_ax) = plt.subplots(1, 2, figsize=(12, 6))

n11,n22=200,200
test_x1 = torch.linspace(3., np.max(x), n11,dtype=torch.float64)
test_x2 = torch.linspace(3., np.max(x), n22,dtype=torch.float64)
test_x = torch.cat([test_x1 , test_x2])
test_index = torch.tensor([True]*n11+[False]*n22)

# Make predictions
with torch.no_grad(), gpytorch.settings.max_cg_iterations(50):
    predictions = likelihood(model(test_x,test_index))
    mean = predictions.mean
    lower, upper = predictions.confidence_region()

# Plot training data as black stars
y1_ax.plot(train_x[:n1].detach().numpy(), train_y[:n1].detach().numpy(), 'k*')
# Predictive mean as blue line
y1x3 = mean[:n11].detach().numpy()[0]
y1_ax.plot(test_x1.detach().numpy(), mean[:n11].detach().numpy(), 'b')
# Shade in confidence
y1_ax.fill_between(test_x1.detach().numpy(), lower[:n11].detach().numpy(), upper[:n11].detach().numpy(), alpha=0.5)
y1_ax.legend(['Observed Values', 'Mean', 'Confidence'])
y1_ax.set_title('Strain energy density')
y1_ax.set_xlabel('$I_1$')
y1_ax.set_ylabel('$W$')

# Plot training data as black stars
y2_ax.plot(train_x[n1:].detach().numpy(), train_y[n1:].detach().numpy(), 'k*')
# Predictive mean as blue line
y2_ax.plot(test_x2.detach().numpy(), mean[n11:].detach().numpy(), 'b')
# Shade in confidence
y2_ax.fill_between(test_x2.detach().numpy(), lower[n11:].detach().numpy(), upper[n11:].detach().numpy(), alpha=0.5)
y2_ax.legend(['Observed Derivatives', 'Mean', 'Confidence'])
y2_ax.set_title('Derivative of strain energy density')
y2_ax.set_xlabel('$I_1$')
y2_ax.set_ylabel('$dW/dI_1$')

plt.show()
