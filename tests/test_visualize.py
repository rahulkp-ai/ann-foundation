import pytest
import numpy as np
import matplotlib
matplotlib.use('Agg')   # non-interactive backend — no display needed
import matplotlib.pyplot as plt
from src.visualize import plot_decision_boundary, plot_training_curve

@pytest.fixture(autouse=True)
def close_plots():
    yield
    plt.close('all')    # clean up after every test

def test_training_curve_returns_figure():
    losses = [1.0, 0.8, 0.6, 0.4, 0.2]
    fig = plot_training_curve(losses)
    assert isinstance(fig, plt.Figure)

def test_training_curve_single_point():
    fig = plot_training_curve([0.5])
    assert isinstance(fig, plt.Figure)

def test_decision_boundary_returns_figure():
    # simple mock model: always returns a Value-like float
    class FakeValue:
        def __init__(self, v): self.data = v

    model = lambda x: FakeValue(x[0] - x[1])   # trivial linear boundary

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([0, 1, -1, 0])
    fig = plot_decision_boundary(model, X, y)
    assert isinstance(fig, plt.Figure)

def test_decision_boundary_title():
    class FakeValue:
        def __init__(self, v): self.data = v
    model = lambda x: FakeValue(0.0)
    X = np.array([[0.0, 0.0], [1.0, 1.0]])
    y = np.array([0, 1])
    fig = plot_decision_boundary(model, X, y, title="Test Title")
    assert fig.axes[0].get_title() == "Test Title"