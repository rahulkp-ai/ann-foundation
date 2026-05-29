import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def make_moons_dataset(n_samples=200, noise=0.1):
    """
    Generates a non-linear dataset and converts labels to -1 and 1 
    to match Tanh activation range.
    """
    X, y = make_moons(n_samples=n_samples, noise=noise)
    # Convert labels from {0, 1} to {-1, 1}
    y = y * 2 - 1
    return X, y

def plot_decision_boundary(model, X, y):
    """
    Plots the decision boundary of the MLP by predicting 
    values over a coordinate grid.
    """
    h = 0.1  # Step size of the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    # Generate a grid of points
    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    # Flatten the grid to pass through the model
    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Predict for every point in the grid
    # Logic: model(list(p)) returns a single Value object 
    # because the output layer has 1 neuron.
    preds = [model(list(p)).data for p in grid]
    
    # Reshape predictions back into the grid shape
    Z = np.array(preds).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    
    # Plot the filled contour (the background colors)
    # Z > 0 identifies the decision boundary for Tanh outputs
    plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap='RdBu')
    
    # Plot the actual data points
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='RdBu', edgecolors='k')
    
    plt.title("Neural Network Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()