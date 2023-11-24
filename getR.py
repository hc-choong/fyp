"""Getting the R and delta squared values from the best trial solutions."""
from variables import V,U,x,T,a,t,N
import numpy as np
from numba import njit, prange

import itertools

Beta= 10**6/(3.17 *T) #calculation of inverse temperature




ki =-np.pi / a
kf= np.pi / a
k_x = np.linspace(ki, kf, num=N+1)
k_y = np.linspace(kf, ki, num=N+1)
k_x = np.delete(k_x,-1)
k_x = k_x + 2*np.pi/(2*N*a)
k_y = np.delete(k_y,-1)
k_y = k_y - 2*np.pi/(2*N*a)
k = np.array([(xi,yi) for yi in k_y for xi in k_x]).reshape(N,N,2)  # to slice k[y-axis,x-axis]
n_by_n = np.array([[x, y] for x, y in itertools.product(range(N), repeat=2)])

# region JIT-able NumPy array functions
@njit(nogil=True, cache=True)
def np_delete_axis0(x: np.array, y: np.array) -> np.array:
    """
    Numba compatible version of np.delete(x, y, axis=0).
    """
    mask = np.ones(len(x), dtype=np.bool8)
    mask[y] = False
    return x[mask,...]


@njit(nogil=True, cache=True)
def np_all_axis1(x: np.array) -> np.array:
    """
    Numba compatible version of np.all(x, axis=1).
    """
    out = np.ones(x.shape[0], dtype=np.bool8)
    for i in range(x.shape[1]):
        out = np.logical_and(out, x[:, i])
    return out
# endregion


# region Math
@njit(nogil=True, cache=True)
def Eta(k: np.array, q: np.array) -> np.float64:
    kx = k[0]
    ky = k[1]
    qx = q[0]
    qy = q[1]
    return np.cos(a*(kx -qx)) - np.cos(a*(ky - qy))


@njit(nogil=True, cache=True)
def Epsilon(q: np.array) -> np.float64:
    qx = q[0]
    qy = q[1]
    return -2*t*(np.cos(a*qx)+np.cos(a*qy))


@njit(nogil=True, cache=True)
def Mu(list_of_q: np.array) -> np.float64:
    sum_E = 0
    for q in range(len(list_of_q)):
        sum_E += Epsilon(list_of_q[q])
    return (1/N * sum_E + 2*x/Beta - 2/Beta)


@njit(nogil=True, cache=True)
def R_val(i: np.int64, j: np.int64, solution: np.array) -> (np.float64, np.float64):
    kpoint = k[j][i]
    delk = solution.reshape(N, N)[j,i]
    Delq = solution.copy().reshape(N, N)
    Delq[j,i] = 0
    q_list = k.reshape(N**2,2)
    qequalsk = np.where(np_all_axis1(x=(q_list == kpoint)))
    q_list = np_delete_axis0(x=q_list, y=qequalsk)
    mu = Mu(q_list)
    R = 0
    for y, x in n_by_n:
        q = k[y][x]
        delta_q = Delq[y,x]
        eta = Eta(k=kpoint,q=q)
        E = Epsilon(q=q) - mu
        lamb = np.sqrt(E**2 + delta_q**2)
        F = np.tanh(Beta*lamb/2)/(4*N*lamb)
        R += (V*(eta**2) - 2*U)*delta_q*F
    delsq = (delk - R)**2
    return R, delsq
# endregion

@njit(nogil=True, cache=True, parallel=True)
def get_R(solution):
    R_array = np.array([0.0 for _ in range(N**2)])
    deltasquared_array = np.array([0.0 for _ in range(N**2)])
    for j in prange(N):
        for i in range(N):
            R, deltasq = R_val(i=i, j=j, solution=solution)
            R_array[i + j*N] = R
            deltasquared_array[i + j*N] = deltasq
    return R_array, deltasquared_array

best_solutions = np.load('best_solutions.npy')

R = np.empty((0, N**2))
DSQ = np.empty((0, N**2))

for sol in best_solutions:
    r, delsq = get_R(sol)
    R = np.vstack((R, r))
    DSQ = np.vstack((DSQ, delsq))

np.save('R.npy',R)
np.save('DSQ.npy',DSQ)