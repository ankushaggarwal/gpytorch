import sys
from model_defs import *
import argparse
import numpy as np
from pymecht import *

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
parser.add_argument("--alpha", dest="alpha",type=float,default=1.,
                  help="Multiply the log probability of Burnouille likelihood by M to regularize the problem", metavar="M")
parser.add_argument("--save-spline", action="store_true", default=False,
                  help="evaluate the model and save as splines")

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
            mm.I1,mm.I4 = I[0],np.array([I[1]])
            dI1,_,_,dI4 = mm.partial_deriv(**params)
            res[i,0] += dI1 if dI1 is not None else 0
            res[i,1] += dI4 if dI4 is not None else 0
    return res

def stress_from_inv(dWdI1,dWdI4,stretches): #returns Cauchy stress
    n = len(dWdI1)
    S = []
    M = np.array([1.,0.,0.])
    for i in range(n):
        F = np.array([[stretches[i,0],0,0],[0,stretches[i,1],0],[0,0,1./stretches[i,0]/stretches[i,1]]])
        s = 2*dWdI1[i]*F@(F.T) + 2*dWdI4[i]*F@(np.outer(M,M))@(F.T)
        s -= s[2,2]*np.eye(3)
        S += [s[0,0],s[1,1]]
    return S

def plot_all(fname='test.png'):
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
        'text.usetex'       : False,
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
    V=np.linspace(0,np.max(data),256)
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
    data = np.abs(dWd_true[:,1]-dWdI4p)/(dWd_true[:,1])*100.
    np.nan_to_num(data,copy=False,posinf=np.nan)
    V=np.linspace(0,100.,256)
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
    from matplotlib.colors import TwoSlopeNorm
    ax = fig.add_axes([x0+0.*dx,y0+2.2*dy,x1,y1])
    data = dW2dI1p
    V=np.linspace(np.min(data),np.max(data),256)
    #V=np.linspace(-150,50,256)
    #CMAP=cm.get_cmap("plasma_r").copy()
    #CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='min')
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap='seismic_r',norm=TwoSlopeNorm(0))
    #CS.cmap.set_under('white')
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
    V=np.linspace(np.min(data),np.max(data),256)
    #V=np.linspace(-150,50,256)
    #CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='min')
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap='seismic_r',norm=TwoSlopeNorm(0))
    #CS.cmap.set_under('white')
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
    plt.savefig(fname,bbox_inches='tight')

def plot_all2(fname='test.png'):
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
        'text.usetex'       : False,
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
    data = dWdI1p
    V=np.linspace(np.min(data),np.max(data),256)
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='max')
    CS.cmap.set_over('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')

    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title('Predicted mean')

    cbaxes = fig.add_axes([x0+dx,y0,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_1$')
    cbar.ax.tick_params(direction='out')

    ################ dWdI4 ##############
    ax = fig.add_axes([x0+2*dx,y0,x1,y1])
    data = dWdI4p
    V=np.linspace(np.min(data),np.max(data),256)
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='max')
    CS.cmap.set_over('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.plot(invs[:,0],invs[:,1],'o',markersize=1,color='black')

    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title('Predicted mean')

    cbaxes = fig.add_axes([x0+3*dx,y0,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_4$')
    cbar.ax.tick_params(direction='out')

    ################ d2W ##############
    from matplotlib.colors import TwoSlopeNorm
    ax = fig.add_axes([x0+0.*dx,y0+1.2*dy,x1,y1])
    data = dW2dI1p
    V=np.linspace(np.min(data),np.max(data),256)
    #V=np.linspace(-150,50,256)
    #CMAP=cm.get_cmap("plasma_r").copy()
    #CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='min')
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap='seismic_r',norm=TwoSlopeNorm(0))
    #CS.cmap.set_under('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title(r'$d^2W/dI_1^2$')

    cbaxes = fig.add_axes([x0+1.*dx,y0+1.2*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$d^2W/dI_1^2$')
    cbar.ax.tick_params(direction='out')

    ax = fig.add_axes([x0+2*dx,y0+1.2*dy,x1,y1])
    data = dW2dI4p
    V=np.linspace(np.min(data),np.max(data),256)
    #V=np.linspace(-150,50,256)
    #CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap=CMAP,extend='min')
    CS = ax.contourf(xv+3,yv+1,data.reshape(xv.shape),V,cmap='seismic_r',norm=TwoSlopeNorm(0))
    #CS.cmap.set_under('white')
    # This is the fix for the white lines between contour levels
    for c in CS.collections:
        c.set_edgecolor("face")
    ax.set_xlabel(r'$I_1$')
    ax.set_ylabel(r'$I_4$')
    ax.set_title(r'$d^2W/dI_4^2$')

    cbaxes = fig.add_axes([x0+3*dx,y0+1.2*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$d^2W/dI_4^2$')
    cbar.ax.tick_params(direction='out')

    ################ dW stddev ##############
    ax = fig.add_axes([x0+0.*dx,y0+2.4*dy,x1,y1])
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

    cbaxes = fig.add_axes([x0+1.*dx,y0+2.4*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_1$ std dev')
    cbar.ax.tick_params(direction='out')

    ax = fig.add_axes([x0+2*dx,y0+2.4*dy,x1,y1])
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

    cbaxes = fig.add_axes([x0+3*dx,y0+2.4*dy,0.015,y1])
    cbar = plt.colorbar(CS,cax=cbaxes)
    cbar.ax.set_ylabel(r'$dW/dI_4$ std dev')
    cbar.ax.tick_params(direction='out')

    ################ Save ##############
    plt.savefig(fname,bbox_inches='tight')

def plot_all3(fname='test2.png'):
    ################# Plotting to compare ##################
    from matplotlib import pyplot as plt
    from matplotlib import rcParams,cm
    from matplotlib.ticker import MultipleLocator, LogLocator
    from scipy.interpolate import interp1d

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
        'text.usetex'       : False,
        'figure.figsize'    : [height*0.8*4, height*2],
        'legend.frameon'    : True,
    }
    rcParams.update(plotParams)
    fig, axs = plt.subplots(2,len(unique_protocols),figsize=(height*0.8*len(unique_protocols), height*2))
    for i,p in enumerate(unique_protocols):
        color=next(axs[0,0]._get_lines.prop_cycler)['color']
        subset = protocols==p
        x = stretches[subset,0]
        if x.max()/x.min()<1.01: #if the stretch in x direction is constant
            x = stretches[subset,1]
        x2 = np.linspace(x.min(),x.max(),100)
        y1 = stressesp_plus[subset,0]
        y2 = stressesp_minus[subset,0]
        axs[0,i].plot(x,stresses[subset,0],'o',color=color)
        axs[0,i].plot(x,stressesp[subset,0],'-',color=color)
        axs[0,i].fill_between(x2,interp1d(x,y1)(x2),interp1d(x,y2)(x2),alpha=0.2)
        axs[1,i].plot(stretches[subset,1],stresses[subset,1],'o',color=color)
        axs[1,i].plot(stretches[subset,1],stressesp[subset,1],'-',color=color)
        x = stretches[subset,1]
        x2 = np.linspace(x.min(),x.max(),100)
        y1 = stressesp_plus[subset,1]
        y2 = stressesp_minus[subset,1]
        axs[1,i].fill_between(x2,interp1d(x,y1)(x2),interp1d(x,y2)(x2),alpha=0.2)
    ################ Save ##############
    plt.savefig(fname,bbox_inches='tight')

    
def save_spline():
    from scipy.interpolate import RectBivariateSpline
    print('Calculating eigendecomposition of the covariance matrix')
    u,v = torch.linalg.eigh(predictions.covariance_matrix)
    print('Eigendecomposition calculated. Saving as splines')
    imp_modes = u/torch.norm(u)>0.01
    imp_eigvals = u[imp_modes].detach().numpy()
    n_modes = len(imp_eigvals)
    imp_vecs = v[:,imp_modes].detach().numpy()
    x2 = x.detach().numpy() + 3
    y2 = y.detach().numpy() + 1
    z = Wp.reshape(n2,n1).T
    mean_sp = RectBivariateSpline(x2,y2,z,s=0)
    eig_sps = []
    for i in range(n_modes):
        eig_sps.append(RectBivariateSpline(x2,y2,imp_vecs[::5,i].reshape(n2,n1).T,s=0))
        #eig_sps.append(RectBivariateSpline(x2,y2,z-imp_vecs[i,::5].reshape(n2,n1).T*np.sqrt(n_modes*imp_eigvals[i]),s=0))
    import pickle
    ff = open('splines.p','wb')
    pickle.dump(x2,ff)
    pickle.dump(y2,ff)
    pickle.dump(mean_sp,ff)
    pickle.dump(u.detach().numpy(),ff)
    pickle.dump(eig_sps,ff)
    ff.close()
