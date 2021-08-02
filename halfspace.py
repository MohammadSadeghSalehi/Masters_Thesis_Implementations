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
def dykstra(v,b,k,x,y):
    x0=x
    for i in range(k):
        x=x+y[:,i]
        x=x+(b[i]-np.dot(v[i,:],x))*v[i,:]
        y[:,i]=y[:,i]+x0-x
        x0=x
    return x, y
def dykstraRev(v,b,k,x,y):
    x0=x
    for i in range(k-1,-1,-1):
        x=x+y[:,i]
        x=x+(b[i]-np.dot(v[i,:],x))*v[i,:]
        y[:,i]=y[:,i]+x0-x
        x0=x
    return x, y
def symDykstra(v,b,k,x,y):
    tempx,tempy=dykstra(v,b,k,x,y)
    x,y= dykstraRev(v,b,k,tempx,tempy)
    return x,y
def anderson(v,b,k,n,x,y):
    # j is number of sequence
    j=5
    p= np.zeros((n,j))
    U= np.zeros((n,j-1))
    for i in range(j):
        p[:,i], y=dykstra(v,b,k,x,y)
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

##__main__
v, b = planeGen(k,n)
cosine=np.max(v@v.T-np.identity(k))
print("Iterations are number of calling Dykstra \n","cosine of samllest angle ",cosine)
###x is Random Starting point
x= np.random.rand(n)*10
z= copy.deepcopy(x)
bound=10e-10

###Dykstra
zDykstra=[]
dyksTol=[]
zDykstra.append(np.linalg.norm(z))
t=time.time()
y=np.zeros((k,n))
z,y=dykstra(v,b,k,z,y)
dyksTol.append(np.abs(np.linalg.norm(z)-zDykstra[-1]))
while np.abs(np.linalg.norm(z)-zDykstra[-1])>bound:
    zDykstra.append(np.linalg.norm(z))
    z,y=dykstra(v,b,k,z,y)
    dyksTol.append(np.abs(np.linalg.norm(z)-zDykstra[-1]))

zDykstra.append(np.linalg.norm(z))
t= time.time()-t
tDykstra=t
print("Dykstra = ",t,"\nIterations = ",len(zDykstra)-1)
z= copy.deepcopy(x)

###Anderson
zAnderson=[]
andTol=[]
zAnderson.append(np.linalg.norm(z))
t=time.time()
y=np.zeros((k,n))
z= anderson(v,b,k,n,z,y)
andTol.append(np.abs(np.linalg.norm(z)-zAnderson[-1]))
while np.abs(np.linalg.norm(z)-zAnderson[-1])>bound:
    zAnderson.append(np.linalg.norm(z))
    z= anderson(v,b,k,n,z,y)
    andTol.append(np.abs(np.linalg.norm(z)-zAnderson[-1]))
zAnderson.append(np.linalg.norm(z))
t= time.time()-t
tAnderson=t
print("Anderson = ",t,"\nIterations = ",(len(zAnderson)-1)*5)
z= copy.deepcopy(x)



fig, (ax1, ax2) = plt.subplots(1, 2)
ax1.plot(np.linspace(0,tDykstra,len(dyksTol)),dyksTol,'--',color='red',label='Dykstra')

ax1.plot(np.linspace(0,tAnderson,len(andTol)),andTol,'--',color='blue',label='Anderson')
ax1.set_yscale('log')
ax1.set(xlabel='Time (sec.)', ylabel='Norm of Difference of two respective iterations')
ax1.grid()
ax1.legend()
ax2.plot(np.linspace(0,len(zDykstra),len(dyksTol)),dyksTol,'--',color='red',label='Dykstra')
ax2.plot(np.linspace(0,len(zAnderson)*5,len(andTol)),andTol,'--',color='blue',label='Anderson')
ax2.set(xlabel='Number of iterations', ylabel='Norm of Difference of two respective iterations')
ax2.set_yscale('log')
ax2.grid()
ax2.legend()
plt.show()
######
