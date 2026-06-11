"""
Visualization utilities for ANN Foundation.
Plots decision boundaries and training curves.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import Callable


def plot_decision_boundary(
    model: Callable,
    X: np.ndarray,
    y: np.ndarray,
    resolution: float = 0.02,
    title: str = "Decision Boundary"
) -> plt.Figure:
    """
    Plot the decision boundary of a trained model.
    
    Args:
        model: Callable that takes a list of floats and returns a Value
        X: Input features, shape (n_samples, 2)
        y: Labels, shape (n_samples,)
        resolution: Grid resolution for boundary plotting
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

    xx, yy = np.meshgrid(
        np.arange(x_min, x_max, resolution),
        np.arange(y_min, y_max, resolution)
    )

    grid_points = np.c_[xx.ravel(), yy.ravel()]
    Z = np.array([model(p.tolist()).data for p in grid_points])
    Z = Z.reshape(xx.shape)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='RdYlGn')
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap='RdYlGn',
                         edgecolors='black', linewidth=0.5, s=60)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    plt.colorbar(scatter, ax=ax, label='Class')
    plt.tight_layout()
    return fig


def plot_training_curve(
    losses: list[float],
    title: str = "Training Loss"
) -> plt.Figure:
    """
    Plot training loss over epochs.
    
    Args:
        losses: List of loss values per epoch
        title: Plot title
    
    Returns:
        matplotlib Figure object
    """
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(losses, color='steelblue', linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(title, fontsize=14)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    return fig