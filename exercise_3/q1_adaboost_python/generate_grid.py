import numpy as np

def generate_grid(X):
    xmax = max(X[:,0])+1
    xmin = min(X[:,0])-1
    ymax = max(X[:,1])+1
    ymin = min(X[:,1])-1

    xmax = max(xmax, ymax)
    ymax = max(abs(ymin), abs(xmin))
    xmax = max(xmax, ymax)
    x = np.arange(-xmax, xmax, .05)
    y = np.arange(-xmax, xmax, .05)
    
    X_grid, Y_grid = np.meshgrid(x,y)
    return [X_grid, Y_grid, x, y]