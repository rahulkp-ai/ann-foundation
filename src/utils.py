# src/utils.py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

def make_moons_dataset(n_samples=200, noise=0.1):
    X, y = make_moons(n_samples=n_samples, noise=noise)
    y = y * 2 - 1  # Map labels from {0, 1} to {-1, 1}
    return X, y

def plot_decision_boundary(model, X, y):
    X = np.asarray(X)
    y = np.asarray(y)
    
    h = 0.1  
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, h),
        np.arange(y_min, y_max, h)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    
    # Evaluate predictions safely extracting the inner numeric value
    preds = []
    for p in grid:
        out = model(list(p))
        # Handle cases where output comes nested inside a list or raw object
        val = out[0].data if isinstance(out, list) else out.data
        preds.append(val)
        
    Z = np.array(preds).reshape(xx.shape)

    plt.figure(figsize=(10, 7))
    plt.contourf(xx, yy, Z > 0, alpha=0.3, cmap='RdBu')
    plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap='RdBu', edgecolors='k')
    plt.title("Neural Network Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.show()