"""Calculates the total error of the solution to the gap equation."""

from variables import V,U,x,T,a,t
import numpy as np
import pygad
from numba import jit
import pylab as plt

N = 8

Beta= 10**6/(3.17 *T) #calculation of inverse temperature

ki = -np.pi/a
kf= np.pi/a
k_x = np.linspace(ki, kf, num=N+1)
k_y = np.linspace(kf, ki, num=N+1)
k_x = np.delete(k_x,-1)
k_x = k_x + 2*np.pi/(2*N*a)
k_y = np.delete(k_y,-1)
k_y = k_y - 2*np.pi/(2*N*a)
k = np.array([(xi,yi) for yi in k_y for xi in k_x]).reshape(N,N,2)  # to slice: k[y-axis,x-axis] to get (kx,ky) because row = y-axis, column = x-axis

# solution = np.array([-0.53253562,  0.52816206,  0.52821604, -0.5324843,   0.52818151, -0.53251904, -0.53255028,  0.52823669,  0.5282061,  -0.53255363, -0.5325134,   0.52822657, -0.53248747,  0.52823632,  0.52820924, -0.53248922]).reshape(N,N)

# solution = np.array([-0.53253562, -0.53253562,  0.52820924,  0.52820924,  0.52823632,  0.5282061, -0.53255363, -0.53255363, -0.53255363, -0.53248922,  0.5282061,   0.5282061,  0.52823632,  0.5282061,  -0.53255363, -0.53255363,  0.52823632,  0.52823632, -0.53255363, -0.53255363, -0.53255363, -0.53248922,  0.5282061,   0.5282061,  0.52823632,  0.52820924, -0.53255363, -0.53255363, -0.53255363, -0.53255363,  0.52823632,  0.5282061,   0.52823632,  0.52823632, -0.53255363, -0.53255363, -0.53255363, -0.53255363,  0.52820924,  0.52823632,  0.5282061,   0.5282061, -0.53248922, -0.53255363, -0.53255363, -0.53255363,  0.52823632,  0.52823632, -0.53255363, -0.53248922,  0.5282061,   0.5282061,   0.52823632,  0.52823632, -0.53248922, -0.53255363, -0.53255363, -0.53255363,  0.52823632,  0.52823632,  0.5282061,   0.52823632, -0.53248922, -0.53248922]).reshape(N,N)


# region JIT-able NumPy array functions
@jit(nopython=True)
def np_delete_axis0(x: np.array, y: np.array) -> np.array:
    """
    Numba compatible version of np.delete(x, y, axis=0).
    """
    mask = np.ones(len(x), dtype=np.bool8)
    mask[y] = False
    return x[mask,...]


@jit(nopython=True)
def np_all_axis1(x: np.array) -> np.array:
    """
    Numba compatible version of np.all(x, axis=1).
    """
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out
# endregion


# eqn
@jit(nopython=True)
def Eta(k: np.array, q: np.array) -> np.float64:
    kx = k[0]
    ky = k[1]
    qx = q[0]
    qy = q[1]
    return np.cos(a*(kx -qx)) - np.cos(a*(ky - qy))


@jit(nopython=True)
def Epsilon(q: np.array) -> np.float64:
    qx = q[0]
    qy = q[1]
    return -2*t*(np.cos(a*qx)+np.cos(a*qy))


@jit(nopython=True)
def Mu(list_of_q: np.array) -> np.float64:
    sum_E = 0
    for q in list_of_q:
        sum_E += Epsilon(q) 
    return (1/N * sum_E + 2*x/Beta - 2/Beta)


@jit(nopython=True)
def DeltaSquared(i: np.int64, j: np.int64, solution: np.array) -> np.float64:
    kpoint = k[j][i]
    delk = solution.reshape(N, N)[j,i]
    Delq = solution.copy().reshape(N, N)
    Delq[j,i] = 0
    q_list = k.reshape(N**2,2)
    qequalsk = np.where(np_all_axis1(x=(q_list == kpoint)))
    q_list = np_delete_axis0(x=q_list, y=qequalsk)
    mu = Mu(q_list)
    R = 0
    for y in range(N):
        for x in range(N):
            q = k[y][x]
            delta_q = Delq[y,x]
            eta = Eta(kpoint,q)
            E = Epsilon(q) - mu
            lamb = np.sqrt(E**2 + delta_q**2)
            F = np.tanh(Beta*lamb/2)/(4*N*lamb)
            R += (V*(eta**2) - 2*U)*delta_q*F
    return (delk - R)**2

loaded_solution = np.load('sol8.npy')

SIGMAsq = 0
for j in range(N):
    for i in range(N):
        sdeltasq = DeltaSquared(i=i, j=j, solution=loaded_solution[15])
        SIGMAsq += sdeltasq

print(SIGMAsq)


# h = 0
# for sol in loaded_solution:
#     SIGMAsq = 0
#     for j in range(N):
#         for i in range(N):
#             sdeltasq = DeltaSquared(i=i, j=j, solution=sol)
#             SIGMAsq += sdeltasq
#     if SIGMAsq < 0.03:
#         h += 1

# print(h)

# print(SIGMAsq)

# Z    = np.array(solution).reshape(N,N)



# plt.imshow(Z,interpolation='none',extent =[-np.pi/a, np.pi/a, -np.pi/a, np.pi/a])
# plt.title("$\Delta_k$")
# plt.xlabel("$k_x$")
# plt.ylabel("$k_y$")
# plt.colorbar() 
# plt.show()