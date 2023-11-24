"""Old version that run faster for 4x4 grid but slower for 8x8 grid"""
from variables import V,U,x,T,a,t,N
import numpy as np
import pygad
from numba import jit
import pylab as plt

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


# region Math
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
# endregion


def fitness_func(ga_instance: pygad.GA, solution: np.array, solution_idx: np.int64) -> np.float64:
    SIGMAsq = 0
    for j in range(N):
        for i in range(N):
            sdeltasq = DeltaSquared(i=i, j=j, solution=solution)
            SIGMAsq += sdeltasq
    fitness = -SIGMAsq
    return fitness

fitness_function = fitness_func

num_genes = N*N
sol_per_pop = int(num_genes*1.5)


num_generations = 30000
num_parents_mating = int(num_genes*3/4)


init_range_low = -4
init_range_high = 4

parent_selection_type = "sss"
keep_parents = int(N**2/16)

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = 5

# parallel_processing= 8

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
                       mutation_percent_genes=mutation_percent_genes
                       )

ga_instance.run()



# # Set your target fitness value
# target_fitness = -1e-06

# # Initialize a variable to track whether the target fitness is reached


# # Run the optimization process in a loop until the target is met
# while True:
#     ga_instance.run()

#     # Get the best solution's fitness value in the current generation
#     best_fitness = np.max(ga_instance.last_generation_fitness)

#     if best_fitness >= target_fitness:
#         break  # Target met

#     # If the target is not met, continue for more generations
#     num_generations += 100  # Increase the number of generations
#     ga_instance.num_generations = num_generations  # Update the GA instance

solution, solution_fitness, solution_idx = ga_instance.best_solution()

print(f"Parameters of the best solution : {solution}")
print(f"Fitness value of the best solution = {solution_fitness}")


ga_instance.plot_fitness()
plt.show()


Z    = np.array(solution).reshape(N,N)



plt.imshow(Z,interpolation='none',extent =[-np.pi/a, np.pi/a, -np.pi/a, np.pi/a])
plt.title("$\Delta_k$")
plt.xlabel("$k_x$")
plt.ylabel("$k_y$")
plt.colorbar() 
plt.show()