# Architecture

## System Overview

ANN Foundation implements two core abstractions:

1. **Value** — a scalar node in a dynamic computation graph
2. **MLP** — a composition of Neuron and Layer objects built on Value

---

## The Value Class

Every scalar in the engine is a `Value` object storing:

| Attribute   | Type       | Purpose                 |
| ----------- | ---------- | ----------------------- |
| `data`      | float      | The scalar value        |
| `grad`      | float      | Gradient ∂L/∂self       |
| `_backward` | Callable   | Local gradient function |
| `_prev`     | set[Value] | Parent nodes (inputs)   |
| `label`     | str        | Debug identifier        |

### Forward Pass

Operations (`+`, `*`, `**`, `tanh`, etc.) create new `Value` nodes
and register a `_backward` closure that knows how to propagate
gradients to the inputs.

Example for multiplication `c = a * b`:

\_backward = lambda: ( a.dict['grad'] += b.data* c.grad, b.dict['grad'] += a.data * c.grad )

### Backward Pass

`backward()` performs a topological sort of the computation graph
and calls each node's `_backward()` in reverse order (from output to inputs).

L.backward()
└─ topological_sort(L)
└─ [L, ab, a, b, c] ← reversed
└─ call \_backward() on each node
This implements reverse-mode automatic differentiation — the same
algorithm underlying PyTorch's autograd.

---

## Gradient Verification

All analytical gradients are verified against numerical differentiation
using the central difference formula:

∂f/∂x ≈ (f(x + h) - f(x - h)) / (2h), h = 1e-5

Maximum allowed error: `1e-5`. Tests fail if any gradient deviates
beyond this threshold.

---

## MLP Architecture

Input → [Neuron × nin] → Layer 1
→ [Neuron × n1] → Layer 2
...
→ [Neuron × nout] → Output
Each `Neuron` computes: `output = activation(w · x + b)`

where `w` and `b` are `Value` objects initialized with `random.uniform(-1, 1)`.

---

## Complexity

| Operation              | Time     | Space      |
| ---------------------- | -------- | ---------- |
| Forward pass (n nodes) | O(n)     | O(n)       |
| Topological sort       | O(V + E) | O(V)       |
| Backward pass          | O(n)     | O(1) extra |

---

## Limitations (by design)

- **Scalar only** — no tensor support. For tensors, see PyTorch.
- **No GPU** — pure Python, CPU only.
- **No batching** — each sample is a separate forward pass.

These are intentional constraints to keep the implementation
readable and educational.
