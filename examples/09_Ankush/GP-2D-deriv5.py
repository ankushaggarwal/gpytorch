#load data
import numpy as np
data = np.load("GOH-derivs.npz")
invs = data['invs']
dWdI1 = data['dWdI1']
dWdI4 = data['dWdI4']
stretches = data['stretches']
stress = data['stress']
nprotocol = 8

def stress_from_inv(dWdI1,dWdI4):
    n = len(dWdI1)
    S = []
    M = np.array([1.,0.,0.])
    for i in range(n):
        F = np.array([[stretches[i,0],0,0],[0,stretches[i,1],0],[0,0,1./stretches[i,0]/stretches[i,1]]])
        s = 2*dWdI1[i]*F@(F.T) + 2*dWdI4[i]*F@(np.outer(M,M))@(F.T)
        s -= s[2,2]*np.eye(3)
        S += [s[0,0],s[1,1]]
    return S

################## GP Part ####################
import torch
import sys
from os.path import dirname, abspath
sys.path.insert(0,dirname(dirname(dirname(abspath(__file__)))))
import gpytorch
train_x = invs.copy()
train_x[:,0] -= 3.
train_x[:,1] -= 1.
#train_x[:,1] *= 10.
#train_x[:,0] *= 10.
train_x = torch.from_numpy(train_x).float()
train_y = torch.vstack((torch.atleast_2d(torch.from_numpy(dWdI1)), torch.atleast_2d(torch.from_numpy(dWdI4)))).T.reshape(-1).float()

train_y += 0.02 * torch.randn(train_y.shape) #add noise
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

torch.save(model.state_dict(), 'model_state.pth')
# Set into eval mode
model.eval()
likelihood.eval()

predictions = likelihood(model(train_x,train_index))
means = predictions.mean.detach().numpy()
dWdI1p = means[::2]
dWdI4p = means[1::2]

stressp = np.array(stress_from_inv(dWdI1p,dWdI4p)).reshape(-1,2)

################# Plotting to compare ##################
from matplotlib import pyplot as plt

#plt.plot(invs[:,0]-3,dWdI1,'o')
#plt.plot(invs[:,1]-1,dWdI4,'o')
#plt.plot(invs[:,0].reshape(nprotocol,-1).T-3,dWdI1p.reshape(nprotocol,-1).T)
#plt.plot(invs[:,1].reshape(nprotocol,-1).T-1,dWdI4p.reshape(nprotocol,-1).T)
#plt.show()

#plt.plot(stress[:,0].reshape(nprotocol,-1).T)
#plt.plot(stressp[:,0].reshape(nprotocol,-1).T)
#plt.show()
#plt.plot(stress[:,1].reshape(nprotocol,-1).T)
#plt.plot(stressp[:,1].reshape(nprotocol,-1).T)
#plt.show()


from mpl_toolkits.mplot3d import Axes3D

# Initialize plot
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(1,2, 1, projection='3d')
ax2 = fig.add_subplot(1,2, 2, projection='3d')
color_idx = np.linspace(0, 1, nprotocol)

for (c,i1,i4,s1,s2,s1p,s2p) in zip(color_idx,invs[:,0].reshape(nprotocol,-1),invs[:,1].reshape(nprotocol,-1),dWdI1.reshape(nprotocol,-1),dWdI4.reshape(nprotocol,-1),dWdI1p.reshape(nprotocol,-1),dWdI4p.reshape(nprotocol,-1)):
        ln1 = ax1.plot(i1, i4, s1,'o',color=plt.cm.cool(c), markersize=2)
        ln2 = ax2.plot(i1, i4, s2,'o',color=plt.cm.cool(c), markersize=2)
        ln3 = ax1.plot(i1, i4, s1p,'-',color=plt.cm.cool(c))
        ln4 = ax2.plot(i1, i4, s2p,'-',color=plt.cm.cool(c))
        
ax1.elev = 20
ax2.elev = 20
ax1.set_xlabel(r'$I_1$')
ax1.set_ylabel(r'$I_4$')
ax1.set_zlabel(r'$\frac{\partial W}{\partial I_1}$')
ax2.set_xlabel(r'$I_1$')
ax2.set_ylabel(r'$I_4$')
ax2.set_zlabel(r'$\frac{\partial W}{\partial I_4}$')

#plt.show()
fig.savefig('results/derivatives.pdf',bbox_inches='tight')

# Initialize plot
fig = plt.figure(figsize=(10, 6))
ax1 = fig.add_subplot(1,2, 1, projection='3d')
ax2 = fig.add_subplot(1,2, 2, projection='3d')
color_idx = np.linspace(0, 1, nprotocol)

for (c,i1,i4,s1,s2,s1p,s2p) in zip(color_idx,stretches[:,0].reshape(nprotocol,-1),stretches[:,1].reshape(nprotocol,-1),stress[:,0].reshape(nprotocol,-1),stress[:,1].reshape(nprotocol,-1),stressp[:,0].reshape(nprotocol,-1),stressp[:,1].reshape(nprotocol,-1)):
        ln1 = ax1.plot(i1, i4, s1,'o',color=plt.cm.cool(c), markersize=2)
        ln2 = ax2.plot(i1, i4, s2,'o',color=plt.cm.cool(c), markersize=2)
        ln3 = ax1.plot(i1, i4, s1p,'-',color=plt.cm.cool(c))
        ln4 = ax2.plot(i1, i4, s2p,'-',color=plt.cm.cool(c))
        
ax1.elev = 20
ax2.elev = 20
ax1.set_xlabel(r'$\lambda_1$')
ax1.set_ylabel(r'$\lambda_2$')
ax1.set_zlabel(r'$\sigma_{11}$')
ax2.set_xlabel(r'$\lambda_1$')
ax2.set_ylabel(r'$\lambda_2$')
ax2.set_zlabel(r'$\sigma_{22}$')

#plt.show()
fig.savefig('results/stresses.pdf',bbox_inches='tight')

################ PREDICTION ##############
#from matplotlib import cm
#n1, n2 = 50, 50
#xv, yv = torch.meshgrid([torch.linspace(0, 0.7, n1), torch.linspace(0, 0.7, n2)])
#test_x = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
#test_index = torch.ones(test_x.shape[0],ndim+1,dtype=bool)
#predictions = likelihood(model(test_x,test_index))
#means = predictions.mean.detach().numpy()
#Wp      = means[::3]
#dWdI1p = means[1::3]
#dWdI4p = means[2::3]
#
#W = np.zeros_like(xv)
#dWdI1 = np.zeros_like(xv)
#dWdI4 = np.zeros_like(xv)
#for i in range(n1):
#    for j in range(n2):
#        w,w1,w4 = 0.,0.,0.
#        for m in mm:
#            m.I1,m.I4=xv[i,j]+3,np.array([yv[i,j]+1])
#            w += m.energy(**params)
#            a,_,_,b = m.partial_deriv(**params)
#            w1 += a
#            if b is not None:
#                w4 += b
#        W[i,j],dWdI1[i,j],dWdI4[i,j] = w,w1,w4
#
#
#Wp = Wp.reshape(xv.shape)
#dWdI1p = dWdI1p.reshape(xv.shape)
#dWdI4p = dWdI4p.reshape(xv.shape)
#Wp -= Wp[0,0]
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(xv.detach().numpy()+3, yv.detach().numpy()+1, Wp, cmap=cm.coolwarm,
#        linewidth=1, antialiased=False)
#surf2 = ax.plot_wireframe(xv.detach().numpy()+3, yv.detach().numpy()+1, W, linewidth=1)
#
#ax.set_xlabel(r'$I_1$')
#ax.set_ylabel(r'$I_4$')
#ax.set_zlabel(r'$\mathcal{W}$')
#ax.elev = 20
##plt.show()
#fig.savefig('results/strain-energy.pdf',bbox_inches='tight')
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(xv.detach().numpy()+3, yv.detach().numpy()+1, np.abs(Wp-W), cmap=cm.coolwarm,
#        linewidth=1, antialiased=False)
#
#ax.set_xlabel(r'$I_1$')
#ax.set_ylabel(r'$I_4$')
#ax.set_zlabel(r'Error')
#ax.elev = 20
##plt.show()
#fig.savefig('results/strain-energy-error.pdf',bbox_inches='tight')
#
#print('Error in W: ',np.linalg.norm(W-Wp)/n1/n2)
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(xv.detach().numpy()+3, yv.detach().numpy()+1, dWdI1p, cmap=cm.coolwarm,
#        linewidth=1, antialiased=False)
#surf2 = ax.plot_wireframe(xv.detach().numpy()+3, yv.detach().numpy()+1, dWdI1, linewidth=1)
#
#ax.set_xlabel(r'$I_1$')
#ax.set_ylabel(r'$I_4$')
#ax.set_zlabel(r'$\frac{\partial W}{\partial I_1}$')
#ax.elev = 20
##plt.show()
#fig.savefig('results/predict-deriv1.pdf',bbox_inches='tight')
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(xv.detach().numpy()+3, yv.detach().numpy()+1, np.abs(dWdI1p-dWdI1), cmap=cm.coolwarm,
#        linewidth=1, antialiased=False)
#
#ax.set_xlabel(r'$I_1$')
#ax.set_ylabel(r'$I_4$')
#ax.set_zlabel(r'Error')
#ax.elev = 20
##plt.show()
#fig.savefig('results/deriv1-error.pdf',bbox_inches='tight')
#
#print('Error in dWdI1: ',np.linalg.norm(dWdI1-dWdI1p)/n1/n2)
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(xv.detach().numpy()+3, yv.detach().numpy()+1, dWdI4p, cmap=cm.coolwarm,
#        linewidth=1, antialiased=False)
#surf2 = ax.plot_wireframe(xv.detach().numpy()+3, yv.detach().numpy()+1, dWdI4, linewidth=1)
#
#ax.set_xlabel(r'$I_1$')
#ax.set_ylabel(r'$I_4$')
#ax.set_zlabel(r'$\frac{\partial W}{\partial I_4}$')
#ax.elev = 20
##plt.show()
#fig.savefig('results/predict-deriv4.pdf',bbox_inches='tight')
#
#fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
#surf = ax.plot_surface(xv.detach().numpy()+3, yv.detach().numpy()+1, np.abs(dWdI4p-dWdI4), cmap=cm.coolwarm,
#        linewidth=1, antialiased=False)
#
#ax.set_xlabel(r'$I_1$')
#ax.set_ylabel(r'$I_4$')
#ax.set_zlabel(r'Error')
#ax.elev = 20
##plt.show()
#fig.savefig('results/deriv4-error.pdf',bbox_inches='tight')
#
#print('Error in dWdI4: ',np.linalg.norm(dWdI4-dWdI4p)/n1/n2)
