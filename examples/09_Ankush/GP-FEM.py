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

########################################################################################
########################################   GPYTORCH PART ###############################
########################################################################################
#GP code
#Import the GPyTorch libraries
import math
import torch
#import gpytorch
#instead import the local gpytorch repo
import importlib.util
spec = importlib.util.spec_from_file_location("gpytorch", "../../gpytorch/__init__.py")
gpytorch = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gpytorch)
print(gpytorch.__file__)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#Convert the numpy arrays to torch tensors
x_data2 = torch.tensor(lam1[:4].flatten())
y_data2 = torch.tensor(s11[:4].flatten())

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(x_data2,y_data2,likelihood)
model.covar_module.base_kernel.lengthscale = 0.5
model.likelihood.noise = 0.05

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},  # Includes GaussianLikelihood parameters
], lr=0.1)

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

training_iter = 50

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(x_data2)
    # Calc loss and backprop gradients
    loss = -mll(output, y_data2)
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


########################################################################################
##################################### FINITE ELEMENT PART ##############################
########################################################################################

def step(Fext,x):
    fglobal = np.zeros(nNodes)
    Kglobal = np.zeros([nNodes,nNodes])

    lambdas = np.zeros(nElem)
    for e in range(nElem):
        lambdas[e] = (x[e+1]-x[e])/Le[e]

    if (any(lambdas<1) or any(lambdas>1.05)):
        print('WARNING: Lambdas are outside the training range')
    stress,dstress = predict_model(lambdas)
    if printout:
        print(lambdas,stress,dstress)
    #assemble the internal force and stiffness matrices
    dNdxi = np.array([-1.,1.])
    for e in range(nElem):
        p,dpdl = stress[e],dstress[e]
        f = dNdxi*p
        K = np.outer(dNdxi,dNdxi)*dpdl/Le[e]
        fglobal[e:e+2] += f
        Kglobal[e:e+2,e:e+2] += K

    #for force acting only on the end
    #fglobal[-1] += factor*Fext

    #for force acting along the length
    for e in range(nElem):
        le = X[e+1]-X[e]
        fglobal[e:e+2] += factor*Fext*le
	
    if printout:
        print(fglobal,x)

    if any(np.isnan(fglobal)):
        if printout:
            print('NaN found in the force vector')
        return 0

    #remove the fixed boundary condition
    fglobal_reduced = np.array(fglobal[np.logical_not(BClogic)]).transpose()

    temp = np.array(Kglobal[np.logical_not(BClogic), :])
    stiff_reduced = np.array(temp[:,np.logical_not(BClogic)])

    #Use Newton's method
    try:
        dx = np.insert(np.linalg.solve(stiff_reduced, -fglobal_reduced),0,0)
    except:
        print('Singular matrix', stiff_reduced)
        dx = 0
    return dx

L=1.
nNodes=50
Fext=0.01
printout=False
factor = -1 #+1 for compression and -1 for extension

nElem = nNodes-1

#uniform node spacing
X = np.linspace(0,L,nNodes)
Le=[X[i+1]-X[i] for i in range(0,nElem)]

BClogic=nNodes*[False]
BClogic[0]=True

MAX=50
TOL=1e-9
x = X.copy()
eps = 1e-4

print("Starting FEM computation")
i=0
while True:
    if printout:
        print('Step:',i)
    dx=step(Fext,x)
    if printout:
        print('dx=',dx)
    x=x+dx
    i += 1
    if np.linalg.norm(dx)<TOL or i>MAX:
        print("converged after ",i," iterations")
        break;

print('Final solution is ',x)
print(np.diff(x)/Le)

Fext=0.05
i=0
while True:
    if printout:
        print('Step:',i)
    dx=step(Fext,x)
    if printout:
        print('dx=',dx)
    x=x+dx
    i += 1
    if np.linalg.norm(dx)<TOL or i>MAX:
        print("converged after ",i," iterations")
        break;

print('Final solution is ',x)
print(np.diff(x)/Le)
