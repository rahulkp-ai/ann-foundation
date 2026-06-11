import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from unittest.mock import patch
from src.utils import make_moons_dataset, plot_decision_boundary


# ── make_moons_dataset ────────────────────────────────────────────────────────

def test_dataset_shape_default():
    X, y = make_moons_dataset()
    assert X.shape == (200, 2)
    assert y.shape == (200,)

def test_dataset_shape_custom():
    X, y = make_moons_dataset(n_samples=100, noise=0.1)
    assert X.shape == (100, 2)
    assert len(y) == 100

def test_dataset_labels_are_minus_one_and_one():
    _, y = make_moons_dataset(n_samples=200)
    unique = set(y)
    assert unique == {-1, 1}

def test_dataset_features_are_2d():
    X, _ = make_moons_dataset(n_samples=50)
    assert X.ndim == 2
    assert X.shape[1] == 2


# ── plot_decision_boundary ────────────────────────────────────────────────────

class FakeValue:
    """Minimal stand-in for engine.Value — has a .data attribute."""
    def __init__(self, v):
        self.data = float(v)

def test_plot_decision_boundary_runs_without_error():
    """Covers lines 12-43 — the grid evaluation and matplotlib calls."""
    model = lambda x: FakeValue(x[0] - x[1])   # trivial linear boundary

    X = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    y = np.array([-1, 1, -1, 1])

    with patch('matplotlib.pyplot.show'):   # suppress plt.show() in tests
        plot_decision_boundary(model, X, y)

    plt.close('all')

def test_plot_handles_list_output():
    """Covers the isinstance(out, list) branch on line 33."""
    model = lambda x: [FakeValue(x[0])]     # returns a list wrapping Value

    X = np.array([[0.5, 0.5], [1.0, 0.0]])
    y = np.array([1, -1])

    with patch('matplotlib.pyplot.show'):
        plot_decision_boundary(model, X, y)

    plt.close('all')