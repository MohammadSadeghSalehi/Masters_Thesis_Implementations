from time import time
import numpy as np
from numpy.core.fromnumeric import reshape
from numpy.linalg.linalg import solve
from PoissonSolver import poisson_solve, laplacian, poisson_solve1, laplacian1
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
np.random.seed(0)

n = 49
np.set_printoptions(linewidth=np.inf)

u0 = np.zeros((n,n))
fOmega1 = np.random.randint(0,100,(n,n))
fOmega2 = np.random.randint(0,100,(n,n))

BCOmega1 = np.zeros((n,n))
BCOmega2 = np.zeros((n,n))


u0Old=u0

u1 = poisson_solve1(n,fOmega1,BCOmega1)
u0 = u1
for i in range (int((n+1)/2)):
    BCOmega2[i,0] = u0[i+int((n-1)/2),int((n-1)/2)]
    BCOmega2[0,i] = u0[int((n-1)/2),i+int((n-1)/2)]

u2 = poisson_solve1(n,fOmega2,BCOmega2)
u0=u2
for i in range (int((n+1)/2)):
    BCOmega1[i+int((n-1)/2),n-1] = u0[i,int((n-1)/2)]
    BCOmega1[n-1,i+int((n-1)/2)] = u0[int((n-1)/2),i]

def MAP(u0,n,fOmega1,fOmega2,BCOmega1,BCOmega2):
    u1 = poisson_solve1(n,fOmega1,BCOmega1)
    u0 = u1
    for i in range (int((n+1)/2)):
        BCOmega2[i,0] = u0[i+int((n-1)/2),int((n-1)/2)]
        BCOmega2[0,i] = u0[int((n-1)/2),i+int((n-1)/2)]

    u2 = poisson_solve1(n,fOmega2,BCOmega2)
    u0 = u2
    for i in range (int((n+1)/2)):
        BCOmega1[i+int((n-1)/2),n-1] = u0[i,int((n-1)/2)]
        BCOmega1[n-1,i+int((n-1)/2)] = u0[int((n-1)/2),i]
    return u0, BCOmega1, BCOmega2


### Acceleration
def anderson(u0,n,f1,f2,BC1,BC2):
    # j is number of sequence
    j=5
    p= np.zeros((n*n,j))
    U= np.zeros((n*n,j-1))
    for i in range(j):
        temp,BC1,BC2=MAP(u0,n,f1,f2,BC1,BC2)
        p[:,i]= np.reshape(temp,n*n)
        u0=np.reshape(p[:,i],(n,n))
    
    ##initializing U
    for i in range(1,j):
        U[:,i-1]=p[:,i]-p[:,i-1]
    
    one = np.ones(j-1)
    #in case of singularity we add "+(10e-20)*np.identity(j-1)" to U^T*U as lambda regularizer
    z=np.linalg.solve(np.matmul(np.transpose(U),U)+(10e-60)*np.identity(j-1),one)
    c=z/np.matmul(np.transpose(z),one)
    final=0
    
    for i in range(j-1):
        final=final+c[i]*p[:,i]

    return np.reshape(final,(n,n)),BC1,BC2



t=time()
mainIter=0
u0Copy=u0.copy()
u0OldCopy=u0Old.copy()
mainNorm = []
mainNorm.append(np.linalg.norm(u0-u0Old))
while np.linalg.norm(u0-u0Old)>10e-20:
    mainIter=mainIter+1
    u0Old = u0
    u0,BCOmega1,BCOmega2 = MAP(u0,n,fOmega1,fOmega2,BCOmega1,BCOmega2)
    mainNorm.append(np.linalg.norm(u0-u0Old))
t=time()-t
timeMain=t
print(timeMain)
print(mainIter)

########Anderson Acceleration
t=time()
u0=u0Copy
u0Old=u0OldCopy
andIter=0
AndersonNorm = []
AndersonNorm.append(np.linalg.norm(u0-u0Old))
while np.linalg.norm(u0-u0Old)>10e-20:
    andIter=andIter+1
    u0Old = u0
    u0,BCOmega1,BCOmega2 = anderson(u0,n,fOmega1,fOmega2,BCOmega1,BCOmega2)
    AndersonNorm.append(np.linalg.norm(u0-u0Old))
t=time()-t
timeAnderson=t
print(timeAnderson)
print(andIter)


##Plot

plt.plot(np.linspace(0,timeMain,len(mainNorm)),mainNorm,'--',color='red',label='Alternating Projections')
plt.plot(np.linspace(0,timeAnderson,len(AndersonNorm)),AndersonNorm,'--',color='blue',label='Anderson Acceleration on AP')
plt.yscale('log')
plt.grid()
plt.ylabel("Norm of Difference of two respective iterations\n (log scale)")
plt.xlabel("Time (sec.)")
plt.legend()
plt.show()