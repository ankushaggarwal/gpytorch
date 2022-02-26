import sys
sys.path.append('../10_Monotonic_GP')
exec(open("../10_Monotonic_GP/common-parts.py").read())

#################################################################################
##################################### Load data #################################
#################################################################################
import pandas as pd
df = pd.read_excel(io='ConstantInvariant_DrAggarwal_022422.xlsx',sheet_name='TVAL100',header=[0,1])
df = df.rename(columns=lambda x: x if not 'Unnamed' in str(x) else '')
subset = df['Protocol'].isin (np.arange(1,8))
df1 = df[subset]
df2 = df1[(df1['Applied']['P11']>0) | (df1['Applied']['P22']>0)]
l1,l2 = df2['Tine']['λ_1'].to_numpy(), df2['Tine']['λ_2'].to_numpy()
P1,P2 = df2['Applied']['P11'].to_numpy(), df2['Applied']['P22'].to_numpy()
stretches = np.vstack((l1,l2)).T
stresses = np.vstack((P1*l1,P2*l2)).T

invs = invariants(stretches)
dWdI1,dWdI4 = partial_derivs(stresses,stretches)
################## GP Part ####################
#################################################################################
################################### Format data #################################
#################################################################################
train_x = invs.copy() 
train_x[:,0] -= 3.
train_x[:,1] -= 1.
train_x = torch.from_numpy(train_x).float()

train_y = torch.vstack((torch.atleast_2d(torch.from_numpy(dWdI1)), torch.atleast_2d(torch.from_numpy(dWdI4))))#.T.reshape(-1).float()

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
train_index0[:,0]=True #choose W=0 at the origin, arbitrarily to remove that problem
train_index0[:,2]=True #the W,4 must be zero at the origin for stress to be zero
train_y0 = torch.tensor([0.,0.])

train_x = torch.vstack((train_x0,train_x))
train_index = torch.vstack((train_index0,train_index))
train_index_w_2nd_derivs = torch.vstack((train_index0,train_index_w_2nd_derivs))
train_y = torch.cat((train_y0,train_y))
train_y_w_2nd_derivs = torch.cat((train_y0,train_y_w_2nd_derivs))

#add uniformly spaced points for enforcing convexity
n1, n2 = 20, 20
xv2, yv2 = torch.meshgrid(torch.linspace(-0.1, torch.max(train_x[:,0]), n1)+0.1, torch.linspace(torch.min(train_x[:,1])-0.1, torch.max(train_x[:,1])+0.1, n2),indexing='xy')
train_x2 = torch.stack([xv2.reshape(n1*n2, 1), yv2.reshape(n1*n2, 1)], -1).squeeze(1)
ndata,ndim = train_x2.shape
train_index2 = torch.zeros(ndata,2*ndim+1,dtype=bool)
train_index2[:,3:]=True
train_y2 = torch.ones_like(train_x2).flatten()

train_x_full = torch.vstack((train_x,train_x2))
#train_index_full =  torch.vstack((train_index_w_2nd_derivs,train_index2))
train_index_full =  torch.vstack((train_index,train_index2))
#train_y_full = torch.cat((train_y_w_2nd_derivs,train_y2))
train_y_full = torch.cat((train_y,train_y2))

#################################################################################
################################ Create model etc.###############################
#################################################################################

likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Interval(1e-5,2e-5))  # Value
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()  # x-derivative 
likelihood3 = gpytorch.likelihoods.GaussianLikelihood()  # y-derivative
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
    combined_likelihood2.alpha = torch.Tensor([args.alpha])
    print('Using a value of nu=',combined_likelihood2.nu[0].item(),' for enforcing monotonicity')
    print('Using a value of small slope=',combined_likelihood2.small_slope[0].item(),' for enforcing monotonicity')
    print('Using a value of alpha=',combined_likelihood2.alpha[0].item(),' for enforcing monotonicity')
    for g in optimizer.param_groups:
        g['lr'] = 0.01
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
    x, y = torch.linspace(torch.min(train_x_full[:,0]), torch.max(train_x_full[:,0]), n1), torch.linspace(torch.min(train_x_full[:,1]), torch.max(train_x_full[:,1]), n2)
    xv, yv = torch.meshgrid(x, y,indexing='xy')
    #xv, yv = torch.meshgrid(torch.linspace(-0.1, torch.max(train_x[:,0]), n1), torch.linspace(torch.min(train_x[:,1])-0.1, torch.max(train_x[:,1]), n2),indexing='xy')
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

    invs2 = test_x.clone().detach()
    invs2[:,0] += 3
    invs2[:,1] += 1

    plot_all2()

    test_x2 = invs.copy() 
    test_x2[:,0] -= 3.
    test_x2[:,1] -= 1.
    test_x2 = torch.from_numpy(test_x2).float()
    ndata,ndim = test_x2.shape
    test_index2 = torch.zeros(ndata,2*ndim+1,dtype=bool)
    test_index2[:,1]=True
    test_index2[:,2]=True
    predictions2 = model(test_x2,x_index = test_index2)
    means2 = predictions2.mean.detach().numpy()
    std_var2 = predictions2.stddev.detach().numpy()
    dWdI1p2 = means2[::2]
    dWdI4p2 = means2[1::2]
    stressesp = np.array(stress_from_inv(dWdI1p2,dWdI4p2,stretches)).reshape(stresses.shape)
    stressesp_plus = np.array(stress_from_inv(dWdI1p2+std_var2[::2],dWdI4p2+std_var2[1::2],stretches)).reshape(stresses.shape)
    stressesp_minus = np.array(stress_from_inv(dWdI1p2-std_var2[::2],dWdI4p2-std_var2[1::2],stretches)).reshape(stresses.shape)

    protocols = df2['Protocol'].to_numpy()
    unique_protocols = set(protocols)
    plot_all3('test2.png')

    #predict rest of the protocols
    subset = (df['Protocol'].isin (np.arange(8,14))) & (df['L/U']==1)
    df1 = df[subset]
    df2 = df1[(df1['Applied']['P11']!=0) & (df1['Applied']['P22']!=0)]
    l1,l2 = df2['Tine']['λ_1'].to_numpy(), df2['Tine']['λ_2'].to_numpy()
    P1,P2 = df2['Applied']['P11'].to_numpy(), df2['Applied']['P22'].to_numpy()
    stretches = np.vstack((l1,l2)).T
    stresses = np.vstack((P1*l1,P2*l2)).T

    invs = invariants(stretches)
    #dWdI1,dWdI4 = partial_derivs(stresses,stretches)

    test_x2 = invs.copy() 
    test_x2[:,0] -= 3.
    test_x2[:,1] -= 1.
    test_x2 = torch.from_numpy(test_x2).float()
    ndata,ndim = test_x2.shape
    test_index2 = torch.zeros(ndata,2*ndim+1,dtype=bool)
    test_index2[:,1]=True
    test_index2[:,2]=True
    predictions2 = model(test_x2,x_index = test_index2)
    means2 = predictions2.mean.detach().numpy()
    std_var2 = predictions2.stddev.detach().numpy()
    dWdI1p2 = means2[::2]
    dWdI4p2 = means2[1::2]
    stressesp = np.array(stress_from_inv(dWdI1p2,dWdI4p2,stretches)).reshape(stresses.shape)
    stressesp_plus = np.array(stress_from_inv(dWdI1p2+std_var2[::2],dWdI4p2+std_var2[1::2],stretches)).reshape(stresses.shape)
    stressesp_minus = np.array(stress_from_inv(dWdI1p2-std_var2[::2],dWdI4p2-std_var2[1::2],stretches)).reshape(stresses.shape)

    protocols = df2['Protocol'].to_numpy()
    unique_protocols = set(protocols)
    plot_all3('test3.png')

    subset = (df['Protocol']>=14) & (df['L/U']==1)
    df1 = df[subset]
    df2 = df1[(df1['Applied']['P11']!=0) & (df1['Applied']['P22']!=0)]
    l1,l2 = df2['Tine']['λ_1'].to_numpy(), df2['Tine']['λ_2'].to_numpy()
    P1,P2 = df2['Applied']['P11'].to_numpy(), df2['Applied']['P22'].to_numpy()
    stretches = np.vstack((l1,l2)).T
    stresses = np.vstack((P1*l1,P2*l2)).T

    invs = invariants(stretches)
    #dWdI1,dWdI4 = partial_derivs(stresses,stretches)

    test_x2 = invs.copy() 
    test_x2[:,0] -= 3.
    test_x2[:,1] -= 1.
    test_x2 = torch.from_numpy(test_x2).float()
    ndata,ndim = test_x2.shape
    test_index2 = torch.zeros(ndata,2*ndim+1,dtype=bool)
    test_index2[:,1]=True
    test_index2[:,2]=True
    predictions2 = model(test_x2,x_index = test_index2)
    means2 = predictions2.mean.detach().numpy()
    std_var2 = predictions2.stddev.detach().numpy()
    dWdI1p2 = means2[::2]
    dWdI4p2 = means2[1::2]
    stressesp = np.array(stress_from_inv(dWdI1p2,dWdI4p2,stretches)).reshape(stresses.shape)
    stressesp_plus = np.array(stress_from_inv(dWdI1p2+std_var2[::2],dWdI4p2+std_var2[1::2],stretches)).reshape(stresses.shape)
    stressesp_minus = np.array(stress_from_inv(dWdI1p2-std_var2[::2],dWdI4p2-std_var2[1::2],stretches)).reshape(stresses.shape)

    protocols = df2['Protocol'].to_numpy()
    unique_protocols = set(protocols)
    plot_all3('test4.png')

    if args.save_spline:
        save_spline()

