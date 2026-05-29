"""
Gradient verification tests for the autograd engine.
Each test compares analytical gradients (from engine.backward())
against numerical gradients (finite difference approximation).
"""
import math
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.engine import Value


# ── Helpers ────────────────────────────────────────────────────────────────

def numerical_grad(f, x_val, h=1e-5):
    """Central difference approximation: (f(x+h) - f(x-h)) / 2h"""
    return (f(x_val + h) - f(x_val - h)) / (2 * h)


def check_grad(val_fn, raw_fn, x_val, tol=1e-5):
    """Run backward and compare to numerical gradient."""
    x = Value(x_val)
    out = val_fn(x)
    out.backward()
    analytical = x.grad
    numerical = numerical_grad(raw_fn, x_val)
    assert abs(analytical - numerical) < tol, (
        f"Gradient mismatch: analytical={analytical:.6f}, numerical={numerical:.6f}"
    )


# ── Basic operations ────────────────────────────────────────────────────────

def test_add_gradient():
    a = Value(2.0)
    b = Value(3.0)
    c = a + b
    c.backward()
    assert abs(a.grad - 1.0) < 1e-6
    assert abs(b.grad - 1.0) < 1e-6


def test_mul_gradient():
    a = Value(4.0)
    b = Value(5.0)
    c = a * b
    c.backward()
    assert abs(a.grad - 5.0) < 1e-6
    assert abs(b.grad - 4.0) < 1e-6


def test_sub_gradient():
    a = Value(7.0)
    b = Value(3.0)
    c = a - b
    c.backward()
    assert abs(a.grad - 1.0) < 1e-6
    assert abs(b.grad - (-1.0)) < 1e-6


def test_div_gradient():
    check_grad(
        lambda x: x / Value(3.0),
        lambda x: x / 3.0,
        x_val=6.0
    )


def test_pow_gradient():
    check_grad(
        lambda x: x ** 3,
        lambda x: x ** 3,
        x_val=2.0
    )


def test_neg_gradient():
    a = Value(5.0)
    b = -a
    b.backward()
    assert abs(a.grad - (-1.0)) < 1e-6


# ── Activation functions ────────────────────────────────────────────────────

def test_tanh_gradient():
    check_grad(
        lambda x: x.tanh(),
        lambda x: math.tanh(x),
        x_val=0.5
    )


def test_relu_gradient_positive():
    check_grad(
        lambda x: x.relu(),
        lambda x: max(0, x),
        x_val=1.5
    )


def test_relu_gradient_negative():
    a = Value(-2.0)
    b = a.relu()
    b.backward()
    assert abs(a.grad - 0.0) < 1e-6


def test_sigmoid_gradient():
    check_grad(
        lambda x: x.sigmoid(),
        lambda x: 1 / (1 + math.exp(-x)),
        x_val=0.8
    )


def test_exp_gradient():
    check_grad(
        lambda x: x.exp(),
        lambda x: math.exp(x),
        x_val=1.0
    )


def test_log_gradient():
    check_grad(
        lambda x: x.log(),
        lambda x: math.log(x),
        x_val=2.0
    )


# ── Composite expressions ────────────────────────────────────────────────────

def test_chain_rule():
    """z = tanh(x * 2 + 1) — tests chain rule through multiple ops."""
    check_grad(
        lambda x: (x * Value(2.0) + Value(1.0)).tanh(),
        lambda x: math.tanh(x * 2 + 1),
        x_val=0.3
    )


def test_gradient_accumulation():
    """x used twice: z = x*x → dz/dx should be 2x."""
    x = Value(3.0)
    z = x * x
    z.backward()
    assert abs(x.grad - 6.0) < 1e-5


def test_shared_node_complex():
    """z = x*y + x — both paths accumulate gradient in x."""
    x = Value(2.0)
    y = Value(3.0)
    z = x * y + x
    z.backward()
    # dz/dx = y + 1 = 4,  dz/dy = x = 2
    assert abs(x.grad - 4.0) < 1e-6
    assert abs(y.grad - 2.0) < 1e-6


def test_radd():
    """Tests __radd__: 3 + Value should work."""
    a = Value(2.0)
    b = 3 + a
    b.backward()
    assert abs(a.grad - 1.0) < 1e-6


def test_rmul():
    """Tests __rmul__: 4 * Value should work."""
    a = Value(3.0)
    b = 4 * a
    b.backward()
    assert abs(a.grad - 4.0) < 1e-6


# ── MLP smoke test ──────────────────────────────────────────────────────────

def test_mlp_backward():
    """MLP forward + backward should not raise and gradients should be non-zero."""
    from src.nn import MLP
    model = MLP(2, [4, 1])
    x = [Value(1.0), Value(-1.0)]
    out = model(x)
    out.backward()
    grads = [abs(p.grad) for p in model.parameters()]
    assert any(g > 0 for g in grads), "All gradients are zero after backward()"


def test_zero_grad():
    """zero_grad() should reset all gradients to 0."""
    from src.nn import MLP
    model = MLP(2, [3, 1])
    x = [Value(0.5), Value(-0.5)]
    out = model(x)
    out.backward()
    model.zero_grad()
    assert all(p.grad == 0.0 for p in model.parameters())
