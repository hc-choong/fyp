"""Animating the best solution from each generation"""
from matplotlib import pyplot as plt
from IPython import display
import numpy as np
from variables import N,a

# Load the best solutions from a file
best_solutions = np.load('best_solutions.npy')

"""Using IPython.display to animate the solutions"""
# for generation, solution in enumerate(best_solutions):
#     sol_arr = np.array(solution)
#     Z    = sol_arr.reshape(N,N)
#     display.clear_output(wait=True)
#     display.display()
#     plt.imshow(Z,interpolation='none',extent =[-np.pi/a, np.pi/a, -np.pi/a, np.pi/a])
#     plt.title("$\Delta_k$ at Generation:"+ str(generation) )
#     plt.xlabel("$k_x$")
#     plt.ylabel("$k_y$")
#     plt.colorbar() 
#     plt.show()


"""HTML5 Video Animation"""
import matplotlib.animation as animation

fig = plt.figure()

def animate(frame):
    sol_arr = np.array(best_solutions[frame])
    Z = sol_arr.reshape(N, N)
    plt.clf()
    plt.imshow(Z, interpolation='none', extent=[-np.pi/a, np.pi/a, -np.pi/a, np.pi/a])
    plt.title("$\Delta_k$ at Generation: " + str(frame))
    plt.xlabel("$k_x$")
    plt.ylabel("$k_y$")
    plt.colorbar()

ani = animation.FuncAnimation(fig, animate, frames=len(best_solutions), interval=200)
# ani.save('animation.html', writer='html')
video = ani.to_html5_video()
html = display.HTML(video)
display.display(html)