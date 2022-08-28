import numpy as np
import sys
ff = open(sys.argv[-1])
alphas,dataloss,loss = [],[],[]
alpha=np.nan
for l in ff.readlines():
    l = l.split(' ')
    if len(l)>0 and l[0]=='Iter':
        #print(l[3],l[4])
        if l[5]=='Dataloss:':
            loss.append(float(l[4]))
            dataloss.append(float(l[6]))
        else:
            loss.append(np.nan)
            dataloss.append(float(l[4]))
        alphas.append(alpha)
    elif len(l)>5 and l[4]=='alpha=':
        alpha=float(l[5])

ff.close()

alphas = np.array(alphas)
dataloss = np.array(dataloss)
loss = np.array(loss)
i = np.arange(len(loss))+1

from matplotlib import pyplot as plt
fig, ax1 = plt.subplots()
ax1.plot(i,loss,label='Loss function',color='red')
ax1.plot(i,dataloss,label='Fit to data',color='blue')
#ax1.semilogy()
ax2 = ax1.twinx()
ax2.plot(i,alphas,label=r'$\alpha$',color='green')
ax2.semilogy()
fig.legend()
#plt.show()
fig.savefig("convergence.png",bbox_inches='tight')
ax1.set_ylim(bottom=np.nanmin(dataloss)-3,top=np.nanmean(dataloss))
fig.savefig("convergence2.png",bbox_inches='tight')
mini = np.nanargmin(dataloss)
print('Minimum data loss was',dataloss[mini],'at iteration',mini,'with alpha=',alphas[mini])
