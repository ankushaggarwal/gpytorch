import sys
from model_defs import *
import argparse

#################################################################################
################################### Load options ################################
#################################################################################
parser = argparse.ArgumentParser(description='Load/train/eval/save GP model')
parser.add_argument("--load-state", dest="state_ifile",
                  help="load model state file", metavar="FILE")
parser.add_argument("--save-state", dest="state_ofile",
                  help="save model state file", metavar="FILE")
parser.add_argument("--n1", dest="n1",type=int,
                  help="train without monotonicity constraints for N1 iterations", metavar="N1")
parser.add_argument("--n2", dest="n2",type=int,
                  help="train with monotonicity constraints for N2 iterations", metavar="N2")
parser.add_argument("--eval", action="store_true", default=False,
                  help="evaluate the model and plot the results")
parser.add_argument("--nu", dest="nu",type=float,default=1.,
                  help="set the monotonicity constraint = NU", metavar="NU")
parser.add_argument("--small-slope", dest="small_slope",type=float,default=0.,
                  help="allow allowable small negative slope upto a value of M", metavar="M")

args = parser.parse_args()

def save_all(fname):
    print('Saving model to state',fname)
    torch.save({
            'GPmodel': model.state_dict(),
            'llh1': likelihood.state_dict(),
            'llh2': likelihood2.state_dict(),
            'llh3': likelihood3.state_dict(),
            'combllh':combined_likelihood.state_dict(),
            'combllh2':combined_likelihood2.state_dict(),
            'optimizer': optimizer.state_dict(),
            }, fname)

def load_all(fname):
    #Load the model etc.
    print('Loading model from state',fname)
    checkpoint = torch.load(fname)
    model.load_state_dict(checkpoint['GPmodel'])
    likelihood.load_state_dict(checkpoint['llh1'])
    likelihood2.load_state_dict(checkpoint['llh2'])
    likelihood3.load_state_dict(checkpoint['llh3'])
    combined_likelihood.load_state_dict(checkpoint['combllh'])
    combined_likelihood2.load_state_dict(checkpoint['combllh2'])
    optimizer.load_state_dict(checkpoint['optimizer'])


#################################################################################
##################################### Create data #################################
#################################################################################

import numpy as np
from pymecht import *

material = MatModel('polyI4','yeoh')
mm = material.models
mm[0].fiber_dirs = [np.array([1,0,0])]
sample = PlanarBiaxialExtension(material,disp_measure='length')
params = sample.parameters
params['c1']=5.
params['c2']=15.
params['c3']=0.
params['c4']=0.
params['d1']=1.
params['d2']=10.
params['d3']=0.

#create stretches for nprotocol protocol angles
deltalmax=1.
npoints=30
stretches = []
nprotocol = 8
times = np.linspace(0.,1.,npoints+1)[1:]

def add_stretches(theta,stretches):
    if theta<pi/4.:
        l1max,l2max = deltalmax,deltalmax*tan(theta)
    else:
        l1max,l2max = deltalmax*tan(pi/2.-theta),deltalmax
    for i in range(npoints):
        stretches += [l1max*times[i],l2max*times[i]]
    return stretches

for theta in np.linspace(0,pi/2.,nprotocol):
    stretches = add_stretches(theta,stretches)

stretches = np.array(stretches).reshape(-1,2)+1
#calculate the stresses
stress = sample.disp_controlled(stretches,params)

def invariants(stretches):
    M = np.array([1,0,0])
    invs = []

    for l1,l2 in stretches:
        F = np.array([[l1,0,0],[0,l2,0],[0,0,1./l1/l2]])
        C = F.T@F
        invs += [np.trace(C),np.dot(M,np.dot(C,M))]

    return np.array(invs).reshape(-1,2)

def partial_derivs(stresses,stretches):
    d1,d2 = [], []
    A = np.zeros([2,2])
    r = np.zeros(2)
    for (s11,s22,lam1,lam2) in np.hstack((stresses,stretches)):
        #print(s11,s22,lam1,lam2)
        A[0,0] = 2*(lam1**2-1./lam1**2/lam2**2)
        A[1,0] = 2*(lam2**2-1./lam1**2/lam2**2)
        A[0,1] = 2*lam1**2 #assumes the fiber direction is the first axis
        r[0],r[1] = s11,s22
        x = np.linalg.solve(A,r)
        d1.append(x[0])
        d2.append(x[1])
    return np.array(d1), np.array(d2)

def partial_derivs_direct(invs):
    res = np.zeros_like(invs)
    for i,I in enumerate(invs):
        for mm in material.models:
            mm.I1,mm.I4 = I[0],I[1]
            dI1,_,_,dI4 = mm.partial_deriv(**params)
            res[i,0] += dI1 if dI1 is not None else 0
            res[i,1] += dI4 if dI4 is not None else 0
    return res

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

invs = invariants(stretches)
dWdI1,dWdI4 = partial_derivs(stress,stretches)

#print(invs,dWdI1,dWdI4,partial_derivs_direct(invs))
################## GP Part ####################
#################################################################################
################################### Format data #################################
#################################################################################
train_x = invs.copy()
train_x[:,0] -= 3.
train_x[:,1] -= 1.
#train_x[:,1] *= 10.
#train_x[:,0] *= 10.
train_x = torch.from_numpy(train_x).float()

train_y = torch.vstack((torch.atleast_2d(torch.from_numpy(dWdI1)), torch.atleast_2d(torch.from_numpy(dWdI4))))#.T.reshape(-1).float()

train_y += 0.02 * torch.randn(train_y.shape) #add noise

#add second derivatives
train_y_w_2nd_derivs = torch.vstack((train_y,torch.ones_like(train_x).T)).T.reshape(-1).float()
train_y = train_y.T.reshape(-1).float()

ndata,ndim = train_x.shape
train_index = torch.empty(ndata,2*ndim+1,dtype=bool)
train_index[:,0]=False
train_index[:,1]=True
train_index[:,2]=True
train_index[:,3]=False
train_index[:,4]=False
train_index_w_2nd_derivs = torch.empty(ndata,2*ndim+1,dtype=bool)
train_index_w_2nd_derivs[:,0]=False
train_index_w_2nd_derivs[:,1]=True
train_index_w_2nd_derivs[:,2]=True
train_index_w_2nd_derivs[:,3]=True
train_index_w_2nd_derivs[:,4]=True

#add a point at the origin
train_x0 = torch.atleast_2d(torch.tensor([0.,0.]))
train_index0 = torch.zeros(1,2*ndim+1,dtype=bool)
train_index0[:,0]=True
train_y0 = torch.tensor([0.])

train_x = torch.vstack((train_x0,train_x))
train_index = torch.vstack((train_index0,train_index))
train_index_w_2nd_derivs = torch.vstack((train_index0,train_index_w_2nd_derivs))
train_y = torch.cat((train_y0,train_y))
train_y_w_2nd_derivs = torch.cat((train_y0,train_y_w_2nd_derivs))

#add uniformly spaced points for enforcing convexity
n1, n2 = 20, 20
xv, yv = torch.meshgrid([torch.linspace(0, torch.max(train_x[:,0]), n1), torch.linspace(torch.min(train_x[:,1]), torch.max(train_x[:,1]), n2)])
train_x2 = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
ndata,ndim = train_x2.shape
train_index2 = torch.zeros(ndata,2*ndim+1,dtype=bool)
train_index2[:,3:]=True
train_y2 = torch.ones_like(train_x2).flatten()

train_x_full = torch.vstack((train_x,train_x2))
train_index_full =  torch.vstack((train_index_w_2nd_derivs,train_index2))
train_y_full = torch.cat((train_y_w_2nd_derivs,train_y2))

#################################################################################
################################ Create model etc.###############################
#################################################################################

likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-5,2e-5))  # Value
likelihood2 = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-5,0.1))  # x-derivative 
likelihood3 = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-5,0.1))  # y-derivative
likelihood4 = gpytorch.likelihoods.BernoulliLikelihood()
likelihood5 = gpytorch.likelihoods.BernoulliLikelihood()
combined_likelihood = CompositeLikelihood((likelihood,likelihood2,likelihood3),train_index)
combined_likelihood2 = CompositeLikelihood((likelihood,likelihood2,likelihood3,likelihood4,likelihood5),train_index_full)

lb1,lb2 = torch.min(train_x,dim=0).values
ub1,ub2 = torch.max(train_x,dim=0).values
model = GPModel(400,ndim,[lb1,lb2],[ub1,ub2])

# Use the adam optimizer
optimizer = torch.optim.Adam([
    {'params': model.parameters()},
    #{'params': likelihood.parameters()},
    {'params': likelihood2.parameters()},
    {'params': likelihood3.parameters()},
    ], lr=0.05)  # Includes GaussianLikelihood parameters

# "Loss" for GPs 
mll = gpytorch.mlls.VariationalELBO(combined_likelihood, model, num_data=train_y.size(0))
mll2 = gpytorch.mlls.VariationalELBO(combined_likelihood2, model, num_data=train_y_full.size(0))

if args.state_ifile is not None:
    load_all(args.state_ifile)

#################################################################################
####################################### TRAIN ###################################
#################################################################################

if args.n1 is not None:
    model.train()
    combined_likelihood.train()

    training_iter = args.n1
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x,x_index=train_index)
        loss = -mll(output, train_y)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f   noise: %.3f %.3f" % (
            i + 1, training_iter, loss.item(),
            likelihood2.noise.item(),
            likelihood3.noise.item(),
        ))
        optimizer.step()

if args.n2 is not None:
    training_iter = args.n2
    #increase the nu and train again with a lower learning rate
    combined_likelihood2.nu = torch.Tensor([args.nu])
    combined_likelihood2.small_slope = torch.Tensor([args.small_slope])
    print('Using a value of nu=',combined_likelihood2.nu[0].item(),' for enforcing monotonicity')
    print('Using a value of small slope=',combined_likelihood2.small_slope[0].item(),' for enforcing monotonicity')
    for g in optimizer.param_groups:
        g['lr'] = 0.0001
    for i in range(training_iter):
        optimizer.zero_grad()
        output = model(train_x_full,x_index=train_index_full)
        output2 = model(train_x,x_index=train_index)
        loss = -mll2(output, train_y_full)
        loss.backward()
        print("Iter %d/%d - Loss: %.3f Dataloss: %.3f   noise: %.3f %.3f" % (
            i + 1, training_iter, loss.item(),
            -mll(output2, train_y).item(),
            likelihood2.noise.item(),
            likelihood3.noise.item(),
        ))
        optimizer.step()

if args.state_ofile is not None:
    save_all(args.state_ofile)

#################################################################################
##################################### EVALUATE ##################################
#################################################################################
if args.eval:
    print('Evaluating the model and plotting the results')
    # Set into eval mode
    model.eval()
    likelihood.eval()

    n1, n2 = 50, 50
    xv, yv = torch.meshgrid([torch.linspace(0, torch.max(train_x[:,0]), n1), torch.linspace(torch.min(train_x[:,1]), torch.max(train_x[:,1]), n2)])
    test_x = torch.stack([xv.reshape(n1*n2, 1), yv.reshape(n1*n2, 1)], -1).squeeze(1)
    test_index = torch.ones(n1*n2,2*ndim+1,dtype=bool)
    predictions = model(test_x,x_index = test_index)
    means = predictions.mean.detach().numpy()
    std_var = predictions.stddev.detach().numpy()
    Wp = means[::5]
    dWdI1p = means[1::5]
    dWdI4p = means[2::5]
    dW2dI1p = means[3::5]
    dW2dI4p = means[4::5]

    #stressp = np.array(stress_from_inv(dWdI1p,dWdI4p)).reshape(-1,2)
    invs2 = test_x.clone().detach()
    invs2[:,0] += 3
    invs2[:,1] += 1
    dWd_true = partial_derivs_direct(invs2)

    ################# Plotting to compare ##################
    from matplotlib import pyplot as plt
    from matplotlib import rcParams,cm
    from matplotlib.ticker import MultipleLocator, LogLocator

    ftsize = 12
    height=2.9 #in inches
    plotParams = {
        # 'backend'           : 'ps',
        'font.family'       : 'serif',
        'font.serif'        : 'Times New Roman',
        'font.size'         : ftsize,
        'axes.labelsize'    : ftsize,
        'legend.fontsize'   : ftsize,
        'xtick.labelsize'   : ftsize-1,
        'ytick.labelsize'   : ftsize-1,
        'lines.markersize'  : 3,
        'lines.linewidth'   : 1,
        'axes.linewidth'    : 1,
        'lines.antialiased' : True,
        'font.size'     : ftsize,
        'text.usetex'       : True,
        'figure.figsize'    : [height*0.8*4, height*2],
        'legend.frameon'    : True,
    }
    rcParams.update(plotParams)

    fig = plt.figure()
    CMAP=cm.get_cmap("plasma").copy()
    errorMap=cm.get_cmap("magma_r").copy()

    x0,y0,x1,y1 = 0.05,0.5,0.2,0.4
    dx,dy = x1+0.0525,-(y1+0.0725)

    ################ dWdI1 ##############
    ax = fig.add_axes([x0,y0,x1,y1])
    data = dWd_true[:,0]
    V=np.linspace(np.min(data),np.max(data),256)
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='max')
    CS.cmap.set_over('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')

    ax.set_ylabel(r'$I_4$')
    ax.set_title('Input model')

    ax = fig.add_axes([x0+dx,y0,x1,y1])
    data = dWdI1p
    CS2 = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),levels=CS.levels,cmap=CMAP,extend='both')
    # This is the fix for the white lines between contour levels
    for c in CS2.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')

    #ax.set_xlabel(r'$I_1$')
    ax.set_title('Predicted mean')

    cbaxes = fig.add_axes([x0+2*dx,y0,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_1$')
    cbar.ax.tick_params(direction='out')

    ax = fig.add_axes([x0+2*dx+0.15,y0,x1,y1])
    data = np.abs(dWd_true[:,0]-dWdI1p)/dWd_true[:,0]*100
    V=np.linspace(0,100,256)
    CS2 = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=errorMap,extend='both')
    # This is the fix for the white lines between contour levels
    for c in CS2.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')
    #ax.set_xlabel(r'$I_1$')
    ax.set_title('Error')

    cbaxes = fig.add_axes([x0+3*dx+0.15,y0,0.015,y1])
    cbar = plt.colorbar(CS2,cax=cbaxes)
    cbar.ax.set_ylabel(r'\% Error')
    cbar.ax.tick_params(direction='out')

    ################ dWdI4 ##############
    ax = fig.add_axes([x0,y0+dy,x1,y1])
    data = dWd_true[:,1]
    V=np.linspace(np.min(data),np.max(data),256)
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='max')
    CS.cmap.set_over('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')

    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    #ax.set_title('Input model')

    ax = fig.add_axes([x0+dx,y0+dy,x1,y1])
    data = dWdI4p
    CS2 = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),levels=CS.levels,cmap=CMAP,extend='both')
    # This is the fix for the white lines between contour levels
    for c in CS2.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')

    ax.set_xlabel(r'$I_1$')
    #ax.set_title('Predicted mean')

    cbaxes = fig.add_axes([x0+2*dx,y0+dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_4$')
    cbar.ax.tick_params(direction='out')

    ax = fig.add_axes([x0+2*dx+0.15,y0+dy,x1,y1])
    data = np.abs(dWd_true[:,1]-dWdI4p)/dWd_true[:,1]*100.
    CS2 = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=errorMap,extend='both')
    # This is the fix for the white lines between contour levels
    for c in CS2.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')
    ax.set_xlabel(r'$I_1$')
    #ax.set_title('Error')

    cbaxes = fig.add_axes([x0+3*dx+0.15,y0+dy,0.015,y1])
    cbar = plt.colorbar(CS2,cax=cbaxes)
    cbar.ax.set_ylabel(r'\% Error')
    cbar.ax.tick_params(direction='out')

    ################ d2W ##############
    ax = fig.add_axes([x0+0.*dx,y0+2.2*dy,x1,y1])
    data = dW2dI1p
    V=np.linspace(0,np.max(data),256)
    CMAP=cm.get_cmap("plasma_r").copy()
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='min')
    CS.cmap.set_under('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title(r'$d^2W/dI_1^2$')

    cbaxes = fig.add_axes([x0+1.*dx,y0+2.2*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$d^2W/dI_1^2$')
    cbar.ax.tick_params(direction='out')

    ax = fig.add_axes([x0+2*dx,y0+2.2*dy,x1,y1])
    data = dW2dI4p
    V=np.linspace(0,np.max(data),256)
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='min')
    CS.cmap.set_under('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title(r'$d^2W/dI_4^2$')

    cbaxes = fig.add_axes([x0+3*dx,y0+2.2*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$d^2W/dI_4^2$')
    cbar.ax.tick_params(direction='out')

    ################ dW stddev ##############
    ax = fig.add_axes([x0+0.*dx,y0+3.4*dy,x1,y1])
    data = std_var[1::5]
    V=np.linspace(np.min(data),np.max(data),256)
    CMAP=cm.get_cmap("plasma_r").copy()
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP)
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')
    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title(r'$dW/dI_1$ std dev')

    cbaxes = fig.add_axes([x0+1.*dx,y0+3.4*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_1$ std dev')
    cbar.ax.tick_params(direction='out')

    ax = fig.add_axes([x0+2*dx,y0+3.4*dy,x1,y1])
    data = std_var[2::5]
    V=np.linspace(np.min(data),np.max(data),256)
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP)
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')
    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title(r'$dW/dI_4$ std dev')

    cbaxes = fig.add_axes([x0+3*dx,y0+3.4*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_4$ std dev')
    cbar.ax.tick_params(direction='out')

    ################ Save ##############
    plt.savefig('test.pdf',bbox_inches='tight')
