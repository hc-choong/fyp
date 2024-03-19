"""Determining the best parameters for the GA"""
from variables import V,U,x,T,a,t,N
import pygad
from numba import njit, prange
import numba
import numpy as np
import itertools



pst = ["sss", "tournament"]

ct = ["single_point", "two_points", "uniform", "scattered"]

mt = ["random","inversion"]

cp = [0.6,0.7]

mp = [0.04,0.08]

mpg = [5,10,15,20]

num_genes = N*N
sol_per_pop = int(num_genes*2)

num_generations = 3000
num_parents_mating = int(num_genes*3/4)

init_range_low = -1.5
init_range_high = 1.5


keep_parents = int(N**2/16)





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


@njit(numba.float64(numba.int64, numba.int64, numba.float64[::1]), nogil=True, cache=True)
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
    for y, x in n_by_n:
        q = k[y][x]
        delta_q = Delq[y,x]
        eta = Eta(k=kpoint,q=q)
        E = Epsilon(q=q) - mu
        lamb = np.sqrt(E**2 + delta_q**2)
        F = np.tanh(Beta*lamb/2)/(4*N*lamb)
        R += (V*(eta**2) - 2*U)*delta_q*F
    return (delk - R)**2
# endregion


@njit(numba.float64(numba.float64[::1]), nogil=True, cache=True, parallel=True)
def ave(sol):
    sol = np.abs(sol)
    average = np.mean(sol)
    return average



@njit(numba.float64(numba.float64[::1]), nogil=True, cache=True, parallel=True)
def get_SigmaSQ(solution: np.array) -> np.float64:
    sigma_sq = 0
    # Don't use the n_by_n array here so that we can run N DeltaSquared() operations in parallel.
    for j in prange(N):
        for i in range(N):
            sdeltasq = DeltaSquared(i=i, j=j, solution=solution)
            sigma_sq += sdeltasq
    return sigma_sq


@njit
def sym(solution):
    sym_diff = 0
    solution = solution.reshape(N, N)
    for i in range(N):
        for j in range(i+1,N):
            dif = solution[i,j] - solution[j,i]
            sym_diff += dif**2 
    return sym_diff


@njit(numba.float64(numba.float64[::1]), nogil=True, cache=True)
def get_fitness(solution: np.array) -> np.float64:
    RMS = np.sqrt(get_SigmaSQ(solution=solution)/N**2)
    return -(RMS/ave(sol=solution))
    # return -(RMS/ave(sol=solution) + sym(solution=solution))



def fitness_func(ga_instance: pygad.GA, solution: np.array, solution_idx: np.int64):
    return get_fitness(solution=solution)


fitness_function = fitness_func

import random
r = random.randint(0,9999)

Q = 3 # number of times to run the GA for each parameter set

# l = np.empty((0,4))
# for p,c,m in itertools.product(psm, cm, mm):
#     u = np.empty((0,4))
#     for _ in range(Q):
#         parent_selection_type = p
#         crossover_type = c
#         mutation_type = m
#         random_seed = r + _ 
#         ga_instance = pygad.GA(num_generations=num_generations,
#                         num_parents_mating=num_parents_mating,
#                         fitness_func=fitness_function,
#                         sol_per_pop=sol_per_pop,
#                         num_genes=num_genes,
#                         init_range_low=init_range_low,
#                         init_range_high=init_range_high,
#                         parent_selection_type=parent_selection_type,
#                         keep_parents=keep_parents,
#                         crossover_type=crossover_type,
#                         mutation_type=mutation_type,
#                         mutation_percent_genes=mutation_percent_genes,
#                         stop_criteria=stop_criteria,
#                         random_seed=random_seed
#                         )
#         ga_instance.run()
#         solution, solution_fitness, solution_idx = ga_instance.best_solution()
#         error = 1 - solution_fitness
#         e = np.array([p,c,m,error])
#         u = np.vstack((u,e))
#     l = np.vstack((l,u))
    
comb = list(itertools.product(pst, ct, mt,cp,mp,mpg))
LLL = [comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))],comb[random.randint(0,len(comb))]]

num = 0
l = np.empty((0,7))
for p,c,m,pc,pm,gpm in LLL:
    u = np.empty((0,7))
    for _ in range(Q):
        parent_selection_type = p
        crossover_type = c
        mutation_type = m
        crossover_probability = pc
        mutation_probability = pm
        mutation_percent_genes = gpm
        random_seed = r + _ 
        ga_instance = pygad.GA(num_generations=num_generations,
                        num_parents_mating=num_parents_mating,
                        fitness_func=fitness_function,
                        sol_per_pop=sol_per_pop,
                        num_genes=num_genes,
                        init_range_low=init_range_low,
                        init_range_high=init_range_high,
                        parent_selection_type=parent_selection_type,
                        keep_parents=keep_parents,
                        crossover_type=crossover_type,
                        mutation_type=mutation_type,
                        mutation_percent_genes=mutation_percent_genes,
                        random_seed=random_seed,
                        crossover_probability=crossover_probability,
                        mutation_probability=mutation_probability
                        )
        ga_instance.run()
        solution, solution_fitness, solution_idx = ga_instance.best_solution()
        error = get_SigmaSQ(solution)
        RMS = np.sqrt(error/N**2)
        RMSperAVE = RMS/ave(solution)
        e = np.array([p,c,m,pc,pm,gpm,RMSperAVE])
        u = np.vstack((u,e))
    num += 1
    print(f"{num} out of {len(pst)*len(ct)*len(mt)*len(cp)*len(mp)*len(mpg)}")
    l = np.vstack((l,u))

l_sorted_asc = l[l[:, 3].argsort()]

import pandas as pd
df = pd.DataFrame(l,columns=["parent_selection_type","crossover_type","mutation_type","crossover_probability","mutation_probability","mutation_percent_genes","rms_per_ave"])
df = df.sort_values(by=["rms_per_ave"])
df.to_csv("test_0.csv",header=False)