from matplotlib import pyplot as plt
import numpy as np
import math

a = 0
b = 1
N = 20


x_uniform = np.linspace(a,b,N)
#x_cheby = (b+a)/2. + (a-b)/2. * np.cos(np.arange(N)*np.pi/(N-1))

def function(x):
    return np.exp(x)* np.cos(8*np.pi*x)

def spectral_laplace_lhs(x):
    N = len(x)
    #a, b = x[0], x[N - 1]
    A = np.zeros((N, N))
    # FIXIT: bestem A_ij
    A[0][0] = 1.0
    A[N - 1][N - 1] = 1.0
    for j in range(1, N - 1):
        for i in range(N):
            A[j][i] = -find_second_derivative_of_lx(x, j, i)
    return A


def find_second_derivative_of_lx(x, j, i):
    N = len(x)
    lx = 0.0
    c1 = 1.0
    for k in range(N):
        for m in range(N):
            secondderivative_lx = 1.0
            if (k != m):
                for n in range(N):
                    if (n != i and n != k and n != m):
                        secondderivative_lx *= (x[j] - x[n])
                lx += secondderivative_lx
    for p in range(N):
        if (p != i):
            c1 *= 1.0 / (x[i] - x[p])
    return lx * c1


def spectral_laplace_rhs(x,ua,ub):
    N = len(x)
    #a,b = x[0],x[N-1]
    B = np.zeros(N)
    # FIXIT: bestem B_i
    my = 2.0
    B[0] = ua
    B[N-1] = ub
    for i in range(1,N-1):
        B[i] = -function(x[i])
    return B
    # set up the spectral method for Laplace eqn and solve the resulting system
def spectral_laplace(x,f):
    ua = 0
    ub = 1
    N=len(x)
    A = spectral_laplace_lhs(x)
    B = spectral_laplace_rhs(x,ua,ub)

    # solve the system

    return np.linalg.solve(A,B)

def error(u):
    e_n=[]
    for i in range (N):
        e_n.append(abs(spectral_laplace(x_uniform,function)- u))
    e_n_max= np.amax(e_n)
    return e_n_max


def find_error(x, function):
    for i in range(len(x)):
        err = error(function(x[i]))
        print (err)

find_error(x_uniform, function)