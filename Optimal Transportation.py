import numpy as np
import torch
import time
from matplotlib import pyplot as plt
torch.set_default_tensor_type(torch.DoubleTensor)
torch.manual_seed(0) # to compare
np.random.seed(0)
#N=400
#N = 625
N = 900
N2 = N**2

# cost matrix
if 0: # use with N=500 and manual_seed(10)
    C = 10*torch.rand((N,N))
    # marginals
    mu = torch.rand(N)
    mu = mu/torch.sum(mu)
    nu = torch.rand(N)
    nu = nu/torch.sum(nu)
else:
## or W2 distance
    L = np.floor(np.sqrt(N)).astype('int')
    M = N//L
    N = L*M
    I = np.arange(N)
    Ix = I//L
    Iy = I%L
    C = (Ix[:,None]-Ix[None,:])**2+(Iy[:,None]-Iy[None,:])**2
    #print (C.shape,C.max())
    C = torch.from_numpy(C)
    #C = torch.clamp(C,max=100)
    # marginals (LxM images)
    Ix = np.reshape(Ix,(M,L))
    Iy = np.reshape(Iy,(M,L))
    c1 = np.random.rand(2)*[M,L]
    c2 = np.random.rand(2)*[M,L]
    dist = np.hypot(c2[0]-c1[0],c2[1]-c1[1])
    mu = np.exp((-(Ix-c1[0])**2 - (Iy-c1[1])**2)/5)
    nu = np.exp((-(Ix-c2[0])**2 - (Iy-c2[1])**2)/5)
    mu /= np.sum(mu)
    nu /= np.sum(nu)
    print ("c1 = ",c1," c2 = ",c2," distance = ",dist)
    plt.figure(1)
    plt.imshow(mu)
    plt.figure(2)
    plt.imshow(nu)
    plt.show()
    mu = torch.from_numpy(np.reshape(mu,N))
    nu = torch.from_numpy(np.reshape(nu,N))

#f = torch.zeros(N)
#g = f.clone()

def project(X0,mu,nu): # X is N times N
    #f = torch.zeros(N)
    #g = f.clone()
    # Y = torch.zeros(N,N)
    # one step
    # Project X0 on the marginal constraints
    X0 = X0 + (mu[:,None]-torch.sum(X0,axis=1,keepdim=True)
               + nu[None,:]-torch.sum(X0,axis=0,keepdim=True)
               + (torch.sum(X0)-1)/N)/N

    # first round
    Y = -torch.clamp(X0,max=0)
    f = -torch.sum(Y,axis=1)/N
    g = (-torch.sum(Y,axis=0)-torch.sum(f))/N
    #Xnew = X0 + Y + f[:,None]+g[None,:]

    fold = f
    gold = g
    err = 1
    it = 0
    while err > .01:
        it = it+1
        beta = it/(it+5)
        f_ = f + beta*(f-fold)
        g_ = g + beta*(g-gold)
        #Yold = Y.clone()
        fold = f.clone()
        gold = g.clone()
        #Xold = Xnew.clone()
        Y = -torch.clamp(X0+f_[:,None]+g_[None,:],max=0)
        f = (-torch.sum(Y,axis=1))/N
        g = (-torch.sum(Y,axis=0)-torch.sum(f))/N
        #Xnew = X0 + Y +f[:,None]+g[None,:]
        
        ### error on X
        Xnew = torch.clamp(X0+Y+f[:,None]+g[None,:],min=0)
        err = torch.sum(torch.abs(mu-torch.sum(Xnew,axis=1)))+torch.sum(torch.abs(nu-torch.sum(Xnew,axis=0)))
        #print(err)

        #err = torch.sqrt(torch.sum((Xnew-Xold)**2))
        #err = torch.sum(torch.abs(Y-Yold))
        
        #err = torch.sum(torch.abs(torch.sum(Y-Yold,axis=0))+torch.abs(torch.sum(Y-Yold,axis=1)))        
        
        #print(err,torch.min(Xnew))
        #print(it,err,torch.sum(torch.clamp(-Xnew,min=0)))

    #print(it,torch.min(Xnew))
    #print(torch.sum(torch.abs(torch.sum(Xnew,axis=1)-mu)))
    Xnew = X0 + Y + f[:,None]+g[None,:]
    print(it)
    return Xnew
t=time.time()
X = mu[:,None]*nu[None,:]
tauout = 1/np.sqrt(torch.sum(C**2).cpu().numpy())

Xold = X.clone()

tnest = 0

eps = .01
niter = np.ceil(4/np.sqrt(eps*tauout)).astype(int)
check = 10

# tests
niter = 20
tauout = tauout/500
print (tauout)

for it in range(niter):
    # inner loop
    ot = tnest
    tnest = (1+np.sqrt(1+4*tnest**2))/2
    beta = (ot-1)/tnest
    X_ = X + beta*(X-Xold)
    X = project(X_ - tauout*C,mu,nu)
    
    if it % check == check-1:
        energy = torch.sum(C*torch.clamp(X,min=0))
        feas = torch.sum(torch.clamp(-X,min=0))
        match = torch.sum(torch.abs(mu-torch.sum(X,axis=1)))+torch.sum(torch.abs(nu-torch.sum(X,axis=0)))
        #gap = penergy+lbda*feasp-(denergy-feasd)
        print("it = ", it,"W = ", energy, "feas =", feas, " match =", match)
        #gaps[it//check] = gap

#le = gaps.shape
#ind = check*(1+np.arange(le[0]))-1

#print("cout final :",penergy,"gap : ",gap)
penergy = torch.sum(C*torch.clamp(X,min=0))
feas = torch.sum(torch.clamp(-X,min=0))
X = torch.clamp(X,min=0)
X = X/torch.sum(X)
print("matching :",torch.sum(torch.abs(mu-torch.sum(X,axis=1))),torch.sum(torch.abs(nu-torch.sum(X,axis=0)))," feas :",feas)
print("distance : ",torch.sqrt(penergy),dist," sparsity :",torch.sum(X>0)/N)
t=time.time()-t
print("Total Time: ",t)
#plt.figure(1)
#plt.plot(gaps)
#plt.figure(2)
#plt.loglog(ind,gaps,'r',ind,co/ind,'k')
#plt.show()
