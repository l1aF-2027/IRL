# """
# Convenience functions for plotting 2D scalar functions
# and histograms.
# """
# from mpl_toolkits.mplot3d import Axes3D
# import matplotlib.pyplot as plt
# from matplotlib import cm
# from matplotlib.ticker import LinearLocator, FormatStrFormatter
# import numpy as np

# def plotFunction (f, xRange, yRange, xLabel, yLabel, zLabel) : 
#     """
#     Create a 3D plot of the function 
#     over xRange x yRange.
#     """

#     F = np.vectorize(f)
#     fig = plt.figure()
#     ax = fig.gca(projection='3d')
#     X, Y = np.meshgrid(xRange, yRange)
#     Z = F(X, Y)
#     surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
#                            linewidth=0, antialiased=False)
#     fig.colorbar(surf, shrink=0.5, aspect=5)
#     ax.set_xlabel(xLabel)
#     ax.set_ylabel(yLabel)
#     ax.set_zlabel(zLabel)
#     plt.show()

# def plotHist (samples, xRange, yRange, xLabel, yLabel, zLabel) : 
#     """
#     Visualize a distribution over two random variables.
#     Stolen from :
#         https://matplotlib.org/stable/gallery/mplot3d/hist3d.html
#     """
#     fig = plt.figure()
#     ax = fig.add_subplot(projection='3d')
#     x, y = samples[:, 0], samples[:, 1]
#     xm, xM = xRange.min(), xRange.max()
#     ym, yM = yRange.min(), yRange.max()
#     bins = max(xRange.size, yRange.size)
#     hist, xedges, yedges = np.histogram2d(x, y, bins=bins, range=[[xm, xM], [ym, yM]])
#     xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
#     xpos = xpos.ravel()
#     ypos = ypos.ravel()
#     zpos = 0
#     dx = dy = np.ones_like(zpos)
#     dz = hist.ravel()
#     ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average')
#     ax.set_xlabel(xLabel)
#     ax.set_ylabel(yLabel)
#     ax.set_zlabel(zLabel)
#     plt.show()

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm

def plotFunction(fn, xRange, yRange, xLabel, yLabel, zLabel):
    """
    Create a 3D surface plot of a 2D function.
    
    Args:
        fn: Function that takes two arguments (x, y) and returns a value
        xRange: Array of x values
        yRange: Array of y values
        xLabel: Label for x-axis
        yLabel: Label for y-axis
        zLabel: Label for z-axis
    """
    # Create the mesh grid
    X, Y = np.meshgrid(xRange, yRange)
    
    # Calculate Z values
    Z = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            Z[i, j] = fn(X[i, j], Y[i, j])
    
    # Create figure and 3D axes
    fig = plt.figure(figsize=(10, 8))
    ax = plt.axes(projection='3d')
    
    # Create the surface plot
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                          linewidth=0, antialiased=True)
    
    # Add labels
    ax.set_xlabel(xLabel)
    ax.set_ylabel(yLabel)
    ax.set_zlabel(zLabel)
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    return fig, ax

def plotTrajectory(trajectory, label=None):
    """
    Plot a 2D trajectory.
    
    Args:
        trajectory: List of states
        label: Optional label for the trajectory
    """
    trajectory = np.array(trajectory)
    plt.plot(trajectory[:, 0], trajectory[:, 1], label=label)
    if label:
        plt.legend()
    plt.xlabel('theta1')
    plt.ylabel('theta2')