import numpy as np
from sklearn.preprocessing import normalize
from scipy import optimize
import matplotlib.pyplot as plt
import time
import copy
np.random.seed(0)
k=n=32
def planeGen(k,n):
    v = np.random.rand(k,n)
    v=normalize(v, axis=1, norm='l2')
    b = np.random.rand(k)
    return v ,b
def dykstra(v,b,k,x):
    for i in range(k):
        x=x+(b[i]-np.dot(v[i,:],x))*v[i,:]
    return x
def dykstraRev(v,b,k,x):
    for i in range(k-1,-1,-1):
        x=x+(b[i]-np.dot(v[i,:],x))*v[i,:]
    return x
def symDykstra(v,b,k,x):
    temp=dykstra(v,b,k,x)
    x= dykstraRev(v,b,k,temp)
    return x
def anderson(v,b,k,n,x):
    # j is number of sequence
    j=5
    p= np.zeros((n,j))
    U= np.zeros((n,j-1))
    for i in range(j):
        p[:,i]=dykstra(v,b,k,x)
        x=p[:,i]
    
    ##initializing U
    for i in range(1,j):
        U[:,i-1]=p[:,i]-p[:,i-1]
    
    one = np.ones(j-1)
    #in case of singularity we add "+(10e-20)*np.identity(j-1)" to U^T*U as lambda regularizer
    z=np.linalg.solve(np.matmul(np.transpose(U),U),one)
    c=z/np.matmul(np.transpose(z),one)
    final=0
    
    for i in range(j-1):
        final=final+c[i]*p[:,i]

    return final

def andersonSym(v,b,k,n,x):
    # j is number of sequence
    j=5
    p= np.zeros((n,j))
    U= np.zeros((n,j-1))
    for i in range(j):
        p[:,i]=symDykstra(v,b,k,x)
        x=p[:,i]
    
    ##initializing U
    for i in range(1,j):
        U[:,i-1]=p[:,i]-p[:,i-1]
    
    one = np.ones(j-1)
    #in case of singularity we add "+(10e-20)*np.identity(j-1)" to U^T*U as lambda regularizer
    z=np.linalg.solve(np.matmul(np.transpose(U),U),one)
    c=z/np.matmul(np.transpose(z),one)
    final=0
    
    for i in range(j-1):
        final=final+c[i]*p[:,i]

    return final

########
def dykstraHom(v,k,x):
    for i in range(k):
        x=x-np.dot(v[i,:],x)*v[i,:]
    return x
def dykstraRevHom(v,k,x):
    for i in range(k-1,-1,-1):
        x=x-np.dot(v[i,:],x)*v[i,:]
    return x
def symDykstraHom(v,k,x):
    temp=dykstraHom(v,k,x)
    x= dykstraRevHom(v,k,temp)
    return x
def gradHom(v,k,x):
    xp = symDykstraHom(v,k,x)
    return x-xp
########Debug
def grad(v,b,k,x):
    xp = symDykstra(v,b,k,x)
    return x-xp

def CG(n,v,b,k,x,bound):
    A=gradHom(v,k,x)
    B=grad(v,b,k,np.zeros(n))
    r=A-B
    p=r
    rsold=np.dot(np.transpose(r),r)
    xCG = []
    CGtol=[]
    xCG.append(np.linalg.norm(x))
    while True:
        Ap=gradHom(v,k,p)
        alpha=rsold/np.dot(np.transpose(p),Ap)
        x = x + alpha*p 
        r = r - alpha*Ap
        rsnew = np.dot(np.transpose(r),r)
        CGtol.append(np.abs(np.linalg.norm(x)-xCG[-1]))
        xCG.append(np.linalg.norm(x))
        if np.sqrt(rsnew)<bound:
            break
        p = r + (rsnew/rsold)*p
        ### beta = rsnew/rsold
        rsold=rsnew
        
        
    return x, xCG, CGtol


##__main__
v, b = planeGen(k,n)
cosine=np.max(v@v.T-np.identity(k))
print("Iterations are number of calling Dykstra \n","cosine of samllest angle ",cosine)
###x is Random Starting point
x= np.random.rand(n)*10
z= copy.deepcopy(x)
bound=10e-10

###Anderson
zAnderson=[]
andTol=[]
zAnderson.append(np.linalg.norm(z))
t=time.time()
z= anderson(v,b,k,n,z)
andTol.append(np.abs(np.linalg.norm(z)-zAnderson[-1]))
while np.abs(np.linalg.norm(z)-zAnderson[-1])>bound:
    zAnderson.append(np.linalg.norm(z))
    z= anderson(v,b,k,n,z)
    andTol.append(np.abs(np.linalg.norm(z)-zAnderson[-1]))
zAnderson.append(np.linalg.norm(z))
t= time.time()-t
tAnderson=t
print("Anderson = ",t,"\nIterations = ",(len(zAnderson)-1)*5)
z= copy.deepcopy(x)
###conjugate gradient
t=time.time()
z,zCG, CGtol=CG(n,v,b,k,z,bound)
t= time.time()-t
tCG=t
print("Conjugate Gradient = ",t,"\nIterations = ",(len(zCG))*2)
z= copy.deepcopy(x)
###Dykstra
zDykstra=[]
dyksTol=[]
zDykstra.append(np.linalg.norm(z))
t=time.time()
z=dykstra(v,b,k,z)
dyksTol.append(np.abs(np.linalg.norm(z)-zDykstra[-1]))
while np.abs(np.linalg.norm(z)-zDykstra[-1])>bound:
    zDykstra.append(np.linalg.norm(z))
    z=dykstra(v,b,k,z)
    dyksTol.append(np.abs(np.linalg.norm(z)-zDykstra[-1]))

zDykstra.append(np.linalg.norm(z))
t= time.time()-t
tDykstra=t
print("Dykstra = ",t,"\nIterations = ",len(zDykstra)-1)
z= copy.deepcopy(x)
###symDykstra
zDykstraSym=[]
symDyksTol=[]
zDykstraSym.append(np.linalg.norm(z))
t=time.time()
z=symDykstra(v,b,k,x)
symDyksTol.append(np.abs(np.linalg.norm(z)-zDykstraSym[-1]))

while np.abs(np.linalg.norm(z)-zDykstraSym[-1])>bound:
    zDykstraSym.append(np.linalg.norm(z))
    z=symDykstra(v,b,k,z)
    symDyksTol.append(np.abs(np.linalg.norm(z)-zDykstraSym[-1]))

zDykstraSym.append(np.linalg.norm(z))

t= time.time()-t
tDykstraSym=t
print("Symmetric Dykstra = ",t,"\nIterations = ",(len(zDykstraSym)-1)*2)
z= copy.deepcopy(x)
###AndersonSym
zAndersonSym=[]
zAndersonSym.append(np.linalg.norm(z))
andSymTol=[]
t=time.time()
z=andersonSym(v,b,k,n,x)
andSymTol.append(np.abs(np.linalg.norm(z)-zAndersonSym[-1]))

while np.abs(np.linalg.norm(z)-zAndersonSym[-1])>bound:
    zAndersonSym.append(np.linalg.norm(z))
    z= andersonSym(v,b,k,n,z)
    andSymTol.append(np.abs(np.linalg.norm(z)-zAndersonSym[-1]))

zAndersonSym.append(np.linalg.norm(z))
t= time.time()-t
tAndersonSym=t
print("Symmetric Anderson = ",t,"\nIterations = ",(len(zAndersonSym)-1)*10)
###plot
fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.linspace(0,tDykstra,len(dyksTol)),dyksTol,'--',color='red',label='Dykstra')
ax1.plot(np.linspace(0,tDykstraSym,len(symDyksTol)),symDyksTol,'--',color='orange',label='Symmetric Dykstra')
ax1.plot(np.linspace(0,tAndersonSym,len(andSymTol)),andSymTol,'--',color='pink',label='Symmetric Anderson')
ax1.plot(np.linspace(0,tAnderson,len(andTol)),andTol,'--',color='blue',label='Anderson')
ax1.plot(np.linspace(0,tCG,len(CGtol)),CGtol,'*-',color='cyan',label='Conjugate gradient')
ax1.set_yscale('log')
ax1.set(xlabel='Time (sec.)', ylabel='Norm of Difference of two respective iterations')
ax1.grid()
ax1.legend()
#ax1.text(tDykstra/3,zAnderson[0]/2,'red=Dykstra,\n blue=Anderson,\ngreen= CG edited, \nyellow= symDykstra,\n pink=SymAnderson')
ax2.plot(np.linspace(0,len(zDykstra),len(dyksTol)),dyksTol,'--',color='red',label='Dykstra')
ax2.plot(np.linspace(0,len(zDykstraSym)*2,len(symDyksTol)),symDyksTol,'--',color='orange',label='Symmetric Dykstra')
ax2.plot(np.linspace(0,len(zAndersonSym)*10,len(andSymTol)),andSymTol,'--',color='pink',label='Symmetric Anderson')
ax2.plot(np.linspace(0,len(zAnderson)*5,len(andTol)),andTol,'--',color='blue',label='Anderson')
ax2.plot(np.linspace(0,(len(zCG))*2,len(CGtol)),CGtol,'--',color='cyan',label='CG')
ax2.set(xlabel='Number of iterations', ylabel='Norm of Difference of two respective iterations')
ax2.set_yscale('log')
ax2.grid()
ax2.legend()
plt.show()