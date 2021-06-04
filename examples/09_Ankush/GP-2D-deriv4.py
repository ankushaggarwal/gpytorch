import torch
import sys
from os.path import dirname, abspath
sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
import gpytorch
import math
from matplotlib import cm
from matplotlib import pyplot as plt
import numpy as np


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

train_x=torch.vstack((torch.atleast_2d(torch.from_numpy(10*(I1.flatten()-3.))),torch.atleast_2d(torch.from_numpy(10*(I4.flatten()-1.))))).T.float()
train_y=torch.vstack((torch.atleast_2d(torch.from_numpy(dWdI1.flatten())),torch.atleast_2d(torch.from_numpy(dWdI4.flatten())))).T.reshape(-1).float()

train_y=(train_x**2).reshape(-1).float()

ndata,ndim = train_x.shape
train_index = torch.empty(ndata,ndim+1,dtype=bool)
train_index[:,0]=False
train_index[:,1:]=True

class LinearMeanGrad(gpytorch.means.Mean):
    def __init__(self, input_size, batch_shape=torch.Size(), bias=True):
        super().__init__()
        self.dim = input_size
        self.register_parameter(name="weights", parameter=torch.nn.Parameter(torch.randn(*batch_shape, input_size, 1)))
        if bias:
            self.register_parameter(name="bias", parameter=torch.nn.Parameter(torch.randn(*batch_shape, 1)))
        else:
            self.bias = None

    def forward(self, x):
        res = x.matmul(self.weights)
        if self.bias is not None:
            res = res + self.bias
        dres = self.weights.expand(self.dim,x.shape[-2]).T #will not work for batches
        return torch.hstack((res,dres))

class GPModelWithDerivatives(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(GPModelWithDerivatives, self).__init__(train_x, train_y, likelihood)
        #self.mean_module = gpytorch.means.ConstantMeanGrad()
        self.mean_module = LinearMeanGrad(2,bias=False)
        self.base_kernel = gpytorch.kernels.RBFKernelGrad(ard_num_dims=2)
        self.covar_module = gpytorch.kernels.ScaleKernel(self.base_kernel)

    def forward(self, x, index):
        index = index.reshape(-1)
        mean_x = self.mean_module(x).reshape(-1)[index]
        full_kernel = self.covar_module(x)
        covar_x = full_kernel[..., index,:][...,:,index]
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

likelihood = gpytorch.likelihoods.GaussianLikelihood()  # Value + x-derivative + y-derivative
model = GPModelWithDerivatives((train_x,train_index), train_y, likelihood)

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50


# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.05)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    optimizer.zero_grad()
    output = model(train_x,train_index)
    loss = -mll(output, train_y)
    loss.backward()
    print("Iter %d/%d - Loss: %.3f   lengthscales: %.3f, %.3f   noise: %.3f" % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.squeeze()[0],
        model.covar_module.base_kernel.lengthscale.squeeze()[1],
        model.likelihood.noise.item()
    ))
    optimizer.step()

# Set into eval mode
model.eval()
likelihood.eval()

predictions = likelihood(model(train_x,train_index))
means = predictions.mean.detach().numpy()
dWdI1p = means[::2].reshape(I1.shape)
dWdI4p = means[1::2].reshape(I4.shape)
#For plotting
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Initialize plot
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(2,1, 1, projection='3d')
ax2 = fig.add_subplot(2,1, 2, projection='3d')
color_idx = np.linspace(0, 1, len(I1))

for (c,i1,i4,s1,s2,s1p,s2p) in zip(color_idx,I1,I4,dWdI1,dWdI4,dWdI1p,dWdI4p):
        #ln1 = ax1.plot(i1, i4, s1,'o',color=plt.cm.cool(c))
        #ln2 = ax2.plot(i1, i4, s2,'o',color=plt.cm.cool(c))
        ln3 = ax1.plot(i1, i4, s1p,'-',color=plt.cm.cool(c))
        ln4 = ax2.plot(i1, i4, s2p,'-',color=plt.cm.cool(c))
        
ax1.elev = 60
ax2.elev = 60
ax1.set_xlabel(r'$I_1$')
ax1.set_ylabel(r'$I_4$')
ax1.set_zlabel(r'$\frac{\partial W}{\partial I_1}$')
ax2.set_xlabel(r'$I_1$')
ax2.set_ylabel(r'$I_4$')
ax2.set_zlabel(r'$\frac{\partial W}{\partial I_4}$')

plt.show()


