import numpy as np

def poisson_solve(m,n,f,BC):
    K = np.zeros(((m-2)*(n-2),(m-2)*(n-2)))
    for i in range(1,m-3):
        for j in range(1,n-3):
            K[i*(n-2)+j, (i+1)*(n-2)+j] = 1 #v_i+1,j
            K[i*(n-2)+j, (i-1)*(n-2)+j] = 1
            K[i*(n-2)+j, i*(n-2)+j+1] = 1
            K[i*(n-2)+j, i*(n-2)+j-1] = 1
            K[i*(n-2)+j, i*(n-2)+j] = -4
    for i in range(1,m-3):
        K[i*(n-2), (i+1)*(n-2)] = 1
        K[i*(n-2), (i-1)*(n-2)] = 1
        K[i*(n-2), i*(n-2)+1] = 1
        K[i*(n-2), i*(n-2)] = -4
        K[i*(n-2)+n-3, i*(n-2)+n-4] = 1
        K[i*(n-2)+n-3, (i+1)*(n-2)+n-3] = 1
        K[i*(n-2)+n-3, (i-1)*(n-2)+n-3] = 1
        K[i*(n-2)+n-3, i*(n-2)+n-3] = -4
    for j in range(1,n-3):
        K[j, n-2+j] = 1
        K[j, j+1] = 1
        K[j, j-1] = 1
        K[j, j] = -4
        K[(m-3)*(n-2)+j, (m-3)*(n-2)+j+1] = 1
        K[(m-3)*(n-2)+j, (m-3)*(n-2)+j-1] = 1
        K[(m-3)*(n-2)+j, (m-4)*(n-2)+j] = 1
        K[(m-3)*(n-2)+j, (m-3)*(n-2)+j] = -4
    K[0,1] = 1
    K[0,n-2] = 1
    K[0,0] = -4
    K[m-3,m-4] = 1
    K[m-3,n-2+m-3] = 1
    K[m-3,m-3] = -4
    K[(m-3)*(n-2),(m-3)*(n-2)+1] = 1
    K[(m-3)*(n-2),(m-4)*(n-2)] = 1
    K[(m-3)*(n-2),(m-3)*(n-2)] = -4
    K[(m-3)*(n-2)+n-3,(m-4)*(n-2)+n-3] = 1
    K[(m-3)*(n-2)+n-3,(m-3)*(n-2)+n-4] = 1
    K[(m-3)*(n-2)+n-3,(m-3)*(n-2)+n-3] = -4
    f_temp = f[1:m-1,1:n-1]
    for i in range(1,m-3):
        f_temp[i, 0] -= BC[i+1, 0]
        f_temp[i, n-3] -= BC[i+1, n-1]
    for j in range(1,n-3):
        f_temp[0, j] -= BC[0, j+1]
        f_temp[m-3, j] -= BC[m-1, j+1]
    f_temp[0,0] -= BC[0,1]+BC[1,0]
    f_temp[0,n-3] -= BC[0,n-2]+BC[1,n-1]
    f_temp[m-3,0] -= BC[m-1,1]+BC[m-2,0]
    f_temp[m-3,n-3] -= BC[m-1,n-2]+BC[m-2,n-1]
    f_temp = np.reshape(f_temp, ((m-2)*(n-2),))
    sol = np.linalg.solve(K, f_temp)
    sol = np.reshape(sol, (m-2, n-2))
    sol_full = BC.copy()
    sol_full[1:m-1,1:n-1] = sol
    return(sol_full)

def laplacian(m,n,A):
    laplacianA = np.empty((m-2,n-2))
    for i in range(0,m-2):
        for j in range(0,n-2):
            laplacianA[i,j] = A[i,j+1] + A[i+1,j] + A[i+1,j+2] + A[i+2,j+1] - 4*A[i+1,j+1]
    return(laplacianA)

def poisson_solve1(n,f,BC):
    K = np.zeros(((n-2)*(n-2),(n-2)*(n-2)))
    for i in range(1,n-3):
        for j in range(1,n-3):
            K[i*(n-2)+j, (i+1)*(n-2)+j] = 1 #v_i+1,j
            K[i*(n-2)+j, (i-1)*(n-2)+j] = 1
            K[i*(n-2)+j, i*(n-2)+j+1] = 1
            K[i*(n-2)+j, i*(n-2)+j-1] = 1
            K[i*(n-2)+j, i*(n-2)+j] = -4
    for i in range(1,n-3):
        K[i*(n-2), (i+1)*(n-2)] = 1
        K[i*(n-2), (i-1)*(n-2)] = 1
        K[i*(n-2), i*(n-2)+1] = 1
        K[i*(n-2), i*(n-2)] = -4
        K[i*(n-2)+n-3, i*(n-2)+n-4] = 1
        K[i*(n-2)+n-3, (i+1)*(n-2)+n-3] = 1
        K[i*(n-2)+n-3, (i-1)*(n-2)+n-3] = 1
        K[i*(n-2)+n-3, i*(n-2)+n-3] = -4
    for j in range(1,n-3):
        K[j, n-2+j] = 1
        K[j, j+1] = 1
        K[j, j-1] = 1
        K[j, j] = -4
        K[(n-3)*(n-2)+j, (n-3)*(n-2)+j+1] = 1
        K[(n-3)*(n-2)+j, (n-3)*(n-2)+j-1] = 1
        K[(n-3)*(n-2)+j, (n-4)*(n-2)+j] = 1
        K[(n-3)*(n-2)+j, (n-3)*(n-2)+j] = -4
    K[0,1] = 1
    K[0,n-2] = 1
    K[0,0] = -4
    K[n-3,n-4] = 1
    K[n-3,n-2+n-3] = 1
    K[n-3,n-3] = -4
    K[(n-3)*(n-2),(n-3)*(n-2)+1] = 1
    K[(n-3)*(n-2),(n-4)*(n-2)] = 1
    K[(n-3)*(n-2),(n-3)*(n-2)] = -4
    K[(n-3)*(n-2)+n-3,(n-4)*(n-2)+n-3] = 1
    K[(n-3)*(n-2)+n-3,(n-3)*(n-2)+n-4] = 1
    K[(n-3)*(n-2)+n-3,(n-3)*(n-2)+n-3] = -4
    f_temp = (1/(n*n))*f[1:n-1,1:n-1]
    for i in range(1,n-3):
        f_temp[i, 0] -= BC[i+1, 0]
        f_temp[i, n-3] -= BC[i+1, n-1]
    for j in range(1,n-3):
        f_temp[0, j] -= BC[0, j+1]
        f_temp[n-3, j] -= BC[n-1, j+1]
    f_temp[0,0] -= BC[0,1]+BC[1,0]
    f_temp[0,n-3] -= BC[0,n-2]+BC[1,n-1]
    f_temp[n-3,0] -= BC[n-1,1]+BC[n-2,0]
    f_temp[n-3,n-3] -= BC[n-1,n-2]+BC[n-2,n-1]
    f_temp = np.reshape(f_temp, ((n-2)*(n-2),))
    sol = np.linalg.solve(K, f_temp)
    sol = np.reshape(sol, (n-2, n-2))
    sol_full = BC.copy()
    sol_full[1:n-1,1:n-1] = sol
    return(sol_full)

def laplacian1(n,A):
    laplacianA = np.empty((n-2,n-2))
    for i in range(0,n-2):
        for j in range(0,n-2):
            laplacianA[i,j] = n*n*(A[i,j+1] + A[i+1,j] + A[i+1,j+2] + A[i+2,j+1] - 4*A[i+1,j+1])
    return(laplacianA)
