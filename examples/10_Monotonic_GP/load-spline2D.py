import pickle
import numpy as np

ff = open('splines.p','rb')
I1 = pickle.load(ff)
I4 = pickle.load(ff)
mean_sp = pickle.load(ff)
imp_eigvals = pickle.load(ff)
eig_sps = pickle.load(ff)

n_modes = len(imp_eigvals)

xv,yv = np.meshgrid(I1,I4)

from matplotlib import pyplot as plt
from matplotlib import rcParams,cm
CMAP=cm.get_cmap("plasma").copy()

data = mean_sp(I1,I4)
V=np.linspace(np.min(data),np.max(data),256)
CS = plt.contourf(xv,yv,data,V,cmap=CMAP)
for c in CS.collections:
    c.set_edgecolor("face")
plt.show()

data = mean_sp(I1,I4,dx=1)
V=np.linspace(np.min(data),np.max(data),256)
CS = plt.contourf(xv,yv,data,V,cmap=CMAP)
for c in CS.collections:
    c.set_edgecolor("face")
plt.show()

data = mean_sp(I1,I4,dy=1)
V=np.linspace(np.min(data),np.max(data),256)
CS = plt.contourf(xv,yv,data,V,cmap=CMAP)
for c in CS.collections:
    c.set_edgecolor("face")
plt.show()

