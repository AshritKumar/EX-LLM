import time
import numpy as np
import matplotlib.pyplot as plt

import sympy as sym

sx,sy = sym.symbols('sx,sy')
# Main function to minimize (Rosenbrock-like function)
sZ = 3*(1-sx)**2 * sym.exp(-(sx**2) - (sy+1)**2) \
      - 10*(sx/5 - sx**3 - sy**5) * sym.exp(-sx**2-sy**2) \
      - 1/3*sym.exp(-(sx+1)**2 - sy**2)


# sym.pprint(sZ)

df_x = sym.diff(sZ, sx)
df_y = sym.diff(sZ, sy)

partial_x_func_lamb = sym.lambdify((sx, sy), df_x, 'numpy')
partial_y_func_lamb = sym.lambdify((sx, sy), df_y, 'numpy')

print("df_x at 1,1", df_x.subs({sx: 1, sy: 1}).evalf())
print("df_y at 1,1", df_y.subs({sx: 1, sy: 1}).evalf())



def show2DPlotManualItrWithoutLambdify():
    start = time.time()
    x_vals1 = np.linspace(-3,3,50)  # takes 4.6 seconds with 50x50 grid
    y_vals1 = np.linspace(-3,3,50)
    # Manual implementation without lambdify
    Z = np.zeros((len(y_vals1), len(x_vals1)))
    for i in range(len(y_vals1)):
        for j in range(len(x_vals1)):
            Z[i, j] = sZ.subs({sx: x_vals1[j], sy: y_vals1[i]}).evalf() # this is a very slow operation - O(n^2) substitutions
    end = time.time()
    print(f"Manual evaluation took {end - start:.2f} seconds")
    show(Z, x_vals1, y_vals1)

def show2DPlotWithLambdify(x_vals, y_vals, shouldShow=False):
    start = time.time()
    # Using lambdify for much faster computation
    print("Converting SymPy expression to fast NumPy function...")
    sZ_func = sym.lambdify((sx, sy), sZ, 'numpy')

    # lambdify converts the symbolic expression into a numerical function that can be evaluated efficiently by directly operating on the complete vectors
    # something like below
    # def sZ_func(sx, sy):
    #     return 3*(1-sx)**2 * np.exp(-(sx**2) - (sy+1)**2) - \
    #            10*(sx/5 - sx**3 - sy**5) * np.exp(-sx**2-sy**2) - \
    #            1/3*np.exp(-(sx+1)**2 - sy**2)

    X, Y = np.meshgrid(x_vals, y_vals) # This creates a X 201 * 201 2D arrays and Y 201*201 2D array
    Z = sZ_func(X, Y)
    end = time.time()
    print(f"Lambdify evaluation took {end - start:.2f} seconds")
    if shouldShow:
        show(Z, x_vals, y_vals)
    return Z

# lr_type vals [fixed, derivative_magnitude, training_epoch]
def applyGradientDescent2D(x_vals, y_vals, lr_type="fixed"):
    start = time.time()
    local_min = np.random.rand(2) # this generated [x,y] where both x,y are in range [0,1)
    local_min = local_min * 4. - 2  # maps to range [-2, 2]
    # local_min = [0, 1.4] # would go to a wrong local min (global min is around [0.296, 0.320])
    # local_min = [ 0.7, -0.9]  # leads to correct global minma
    starting_point = local_min.copy()
    print(f"Starting gradient descent from initial point: {starting_point}")
    learning_rate = 0.01  # Smaller steps for better convergence
    training_epochs = 10000
    modelTrajectory = np.zeros((training_epochs + 1, 2))  # +1 to include starting point
    gradients = []
    lrs = []
    
    # Store the starting point first
    modelTrajectory[0] = local_min.copy()
    
    for i in range(training_epochs):
        # 1 compute the gradient array at local_min point
        grad = computeGraditentsWithLambdify(local_min[0], local_min[1])
        # 2 update the local_min point using gradient descent rule
        lr = learning_rate
        
        # This is time based or training epoch based learning rate, the learning rate gets closer to zero as we progress
        if (lr_type == "training_epoch"):
            lr = learning_rate * (1 - (i+1)/training_epochs)
            # lr = lr * (1 - (i+1)/training_epochs)
        
        # derivative/gradient based learning rate decay - the idea is to reduce learning rate as we get closer to the minimum
        # this helps with stability near the minimum where large steps can cause oscillation
        # This is called "learning rate decay"
        grad_norm = np.linalg.norm(grad)
        if (lr_type == "derivative_magnitude"):
            if grad_norm > 0:
                # lr = lr / (1.0 + grad_norm)
                lr = learning_rate / (1.0 + grad_norm)
                # lr = learning_rate * grad_norm
        
        gradients.append(grad)
        lrs.append(lr)

        # 3. update the local_min point using the computed learning rate and gradient
        local_min = local_min - lr * grad

        # 4. store the updated point in local min trajectory   
        modelTrajectory[i + 1] = local_min.copy()  # Store at i+1 since i=0 is starting point

        # Check for convergence (gradient close to zero)
        grad_magnitude = np.linalg.norm(grad) # np.linalg.norm(grad) this is sqrt(grad[0]^2 + grad[1]^2)
        if grad_magnitude < 0.001:  # Very small gradient = near minimum
            print(f"**** Converged at epoch {i}, gradient magnitude: {grad_magnitude:.6f}")
            modelTrajectory = modelTrajectory[:i+2]  # Trim unused trajectory
            break
    
    print("Final gradient is ", grad)
    print(f"Final gradient magnitude: {np.linalg.norm(grad):.6f}")
    print("WARNING: If gradient magnitude > 0.01, you haven't reached a true minimum!")
    end = time.time()
    print(f"Final point: {local_min}")
    print(f"Final function value: {sZ.subs([(sx, local_min[0]), (sy, local_min[1])]).evalf()}")
    print(f"Gradient descent took {end - start:.2f} seconds")

    print("Plotting the function and model trajectory...")
    # function values over (x_vals, y_vals)
    Z = show2DPlotWithLambdify(x_vals, y_vals, False)
    show2DplotWithLocalMinTrajectory(Z, x_vals, y_vals, modelTrajectory, starting_point, local_min, gradients, lrs, lr_type)
    

def computeGraditentsWithLambdify(x,y):
    return np.array([float(partial_x_func_lamb(x, y)), float(partial_y_func_lamb(x, y))])

def computeGradientsWithoutLambdify(x, y):
    partial_x = df_x.subs([(sx, x), (sy, y)])
    partial_y = df_y.subs([(sx, x), (sy, y)])
    return np.array([float(partial_x), float(partial_y)])

def show2DplotWithLocalMinTrajectory(arr2DVals, x_vals, y_vals, trajectory, starting_point, local_min, gradients, lrs, lr_type):
    # Create subplots: 1 row, 3 columns
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 8))
    
    # Main plot: 2D function with trajectory
    im = ax1.imshow(arr2DVals, extent=[x_vals[0],x_vals[-1],y_vals[0],y_vals[-1]], 
                    vmin=-5, vmax=5, origin='lower', cmap='RdYlBu_r')
    ax1.plot(starting_point[0], starting_point[1], 'bs', label='Starting Point', markersize=8)
    ax1.plot(local_min[0], local_min[1], 'ro', label='Local Minimum', markersize=8)
    ax1.plot(trajectory[:, 0], trajectory[:, 1], 'ko-', label='Gradient Trajectory', linewidth=1, markersize=2, alpha=0.7)
    ax1.legend()
    plt.colorbar(im, ax=ax1, label='Function Value')
    ax1.set_title(f'2D Function Visualization\n(Learning Rate: {lr_type})')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Learning Rate Evolution
    epochs = np.arange(len(lrs))
    ax2.plot(epochs, lrs, 'b-', linewidth=2, alpha=0.8)
    ax2.set_title(f'Learning Rate Evolution\n(Type: {lr_type})')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate')
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(bottom=0)
    
    # Plot 3: Gradient Magnitude Evolution
    grad_magnitudes = np.linalg.norm(gradients, axis=1)
    ax3.plot(epochs, grad_magnitudes, 'r-', linewidth=2, alpha=0.8)
    ax3.set_title(f'Gradient Magnitude Evolution\n(Type: {lr_type})')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Gradient Magnitude')
    ax3.grid(True, alpha=0.3)
    ax3.set_yscale('log')  # Log scale for better visualization
    
    # Add convergence threshold line
    ax3.axhline(y=0.001, color='g', linestyle='--', alpha=0.7, label='Convergence Threshold')
    ax3.legend()
    
    plt.tight_layout()
    print(f"LRs {lrs[:10]}")  # Show first 10 learning rates
    print("Displaying plot with trajectory...")
    plt.show()
  

def show(arr2DVals, x_vals, y_vals):
    # let's have a look!
    plt.figure(figsize=(10, 8))
    im = plt.imshow(arr2DVals, extent=[x_vals[0],x_vals[-1],y_vals[0],y_vals[-1]], 
                    vmin=-5, vmax=5, origin='lower', cmap='RdYlBu_r')

    plt.colorbar(im, label='Function Value')
    plt.xlabel('x')
    plt.ylabel('y') 
    plt.title('2D Function Visualization')
    plt.grid(True, alpha=0.3)
    print("Displaying plot...")
    plt.show()


if __name__ == "__main__":
    x_vals = np.linspace(-3,3,201)  # 201x201 = 40,401 evaluations, takes 0.08 sec because of lambdafy and meshgrid
    y_vals = np.linspace(-3,3,201)
    # print("Running manual evaluation...")
    # show2DPlotManualItrWithoutLambdify()
    # print("Running lambdify evaluation...")
    # Z = show2DPlotWithLambdify(x_vals, y_vals, shouldShow=False)
    # print(Z)
    # print(Z.shape)
    print("Running gradient descent...")
    # lr_type vals [fixed, derivative_magnitude, training_epoch]
    applyGradientDescent2D(x_vals, y_vals, lr_type="training_epoch")



