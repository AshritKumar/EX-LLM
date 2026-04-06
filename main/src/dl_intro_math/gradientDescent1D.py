import numpy as np
import matplotlib.pyplot as plt

import sympy as sym
from sympy.plotting import plot as symplot

import matplotlib_inline.backend_inline
matplotlib_inline.backend_inline.set_matplotlib_formats('svg')

def solve(func, variable, value):
    return func.subs(variable, value)

x = sym.symbols('x')
fx =  3*x**2 - 3*x + 4
dx = sym.diff(fx)

learning_rate = 0.1
training_epochs = 100

def plotFxDx():
    print("Function:")
    sym.pprint(fx)
    print("\nDerivative:")
    sym.pprint(dx)

    print("f(0.5) =", solve(fx, x, 0.5))
    print("f'(0.5) =", solve(dx, x, 0.5))

    p1 = symplot(fx, dx, (x, -2, 2), xlabel='x', ylabel='y', title='Function and its Derivative')
    p1.show()

######################### compute Gradient descent for fx #########################

def compureGD1():
    x_vals = np.linspace(-2, 2, 1000)
    local_min = np.random.choice(x_vals, 1)[0]
    print("Initial local min ", local_min)
    for i in range(training_epochs):
        grad = solve(dx, x, local_min)
        local_min = local_min - grad * learning_rate
        # if abs(grad) < 1e-10:
        #     print(f"Converged at iteration {i}")
        #     break
        if grad == 0:
            print(f"Gradient is zero at iteration {i}")
            break
            
    
    print("Final local min ", local_min)
    # plot results
    y_vals = [solve(fx, x, val) for val in x_vals]
    dy_vals = [solve(dx, x, val) for val in x_vals]
    plt.plot(x_vals, y_vals, x_vals, dy_vals)
    # plot fx value at local min
    plt.plot(local_min, solve(fx, x, local_min), 'ro', markersize=5)
    # plot fx value at local min
    plt.plot(local_min, solve(dx, x, local_min), 'ko', markersize=5)
    plt.xlim(x_vals[[0,-1]]) # first and last index vaues of x vals
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend(['fx', 'dx', 'fx min', 'dx at 0'])
    plt.title(f"Emperical minima {local_min}")
    plt.show()

def compureGD2():
    # store the grad and local_min at each to plot the details
    x_vals = np.linspace(-2, 2, 1000)
    local_min = np.random.choice(x_vals, 1)[0]
    model_parms = np.zeros((training_epochs, 2))
    print("Initial local min ", local_min)
    for i in range(training_epochs):
        grad = solve(dx, x, local_min)
        local_min = local_min - grad * learning_rate
        model_parms[i] = local_min, grad
    
    print("Final local min ", local_min)
    print("Final gradient ", solve(dx, x, local_min))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Function and derivative
    x_vals = np.linspace(-2, 2, 1000)
    y_vals = [solve(fx, x, val) for val in x_vals]
    dy_vals = [solve(dx, x, val) for val in x_vals]
    ax1.plot(x_vals, y_vals, x_vals, dy_vals)
    ax1.plot(local_min, solve(fx, x, local_min), 'ro', markersize=5)
    ax1.plot(local_min, solve(dx, x, local_min), 'ko', markersize=5)
    ax1.set_xlim(x_vals[[0,-1]])
    ax1.set_xlabel('x')
    ax1.set_ylabel('f(x)')
    ax1.legend(['fx', 'dx', 'fx min', 'dx at 0'])
    ax1.set_title(f"Emperical minima {local_min}")
    
    # Plot 2: Training progress - local_min
    ax2.plot(model_parms[:, 0], 'o-', label='local_min')
    ax2.set_xlabel('Iteration')
    ax2.set_ylabel('Local min Value')
    ax2.legend()
    ax2.set_title('Training Progress - Local Min')
    ax2.set_title(f'Initial local_min {model_parms[0][0]:.5f}, Final estimated minimum: {local_min:.5f}')
    
    # Plot 3: Training progress - gradient
    ax3.plot(model_parms[:, 1], 'o-', label='gradient')
    ax3.set_xlabel('Iteration')
    ax3.set_ylabel('Gradient Value')
    ax3.legend()
    ax3.set_title('Training Progress - Gradient')
    
    plt.tight_layout()
    plt.show()


compureGD2()

