---
title: ANN Foundation
emoji: 🧠
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 4.0.0
app_file: app.py
pinned: false
license: mit
short_description: Autograd engine and MLP built from scratch in pure Python
tags:
  - deep-learning
  - machine-learning
  - autograd
  - neural-network
  - education
  - python
---

# ANN Foundation

> A minimal autograd engine and neural network library built from scratch in pure Python — no PyTorch, no TensorFlow.

[![Python](https://img.shields.io/badge/Python-3.8%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![CI](https://github.com/rahulkp-ai/ann-foundation/actions/workflows/ci.yml/badge.svg)](https://github.com/rahulkp-ai/ann-foundation/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rahulkp-ai/ann-foundation/branch/main/graph/badge.svg)](https://codecov.io/gh/rahulkp-ai/ann-foundation)
[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/rahulkp-ai/ann-foundation)

---

## Overview

ANN Foundation implements the complete neural network learning pipeline from mathematical
first principles. Every component — scalar arithmetic, automatic differentiation,
gradient accumulation, and the training loop — is written in pure Python with no
deep learning framework dependencies.

```
Forward Pass  →  Loss Computation  →  Backward Pass  →  Gradient Descent  →  Repeat
```

The goal is not to replace PyTorch. The goal is to understand exactly what PyTorch
is doing under the hood — by building it yourself.

---

## How It Works

The autograd engine builds a dynamic computation graph during the forward pass.
Each `Value` node stores its scalar data, accumulated gradient, and a `_backward`
closure that propagates gradients to its parent nodes via the chain rule.

![Computation Graph](assets/computation_graph.png)

> **Panel A** — Expression `L = a × b + a`: the forward pass builds the graph,
> `backward()` traverses it in reverse topological order. Note that `a` feeds into
> two branches — gradients accumulate correctly: `grad(a) = 3.0 + 1.0 = 4.0`.
>
> **Panel B** — Single neuron `h = tanh(w·x + b)`: the chain rule flows from
> output back through activation → addition → multiplication, producing exact
> analytical gradients verified against numerical differentiation.

---

## Features

- **Autograd engine** — reverse-mode automatic differentiation via topological sort
- **Backpropagation** — correct gradient accumulation with chain rule
- **Activation functions** — `tanh`, `relu`, `sigmoid` with verified analytical gradients
- **Operators** — `+`, `-`, `*`, `/`, `**` and their reverse variants (`radd`, `rmul`, etc.)
- **MLP framework** — composable `Neuron → Layer → MLP` with configurable depth and activation
- **Training loop** — forward pass, loss, backward, gradient descent
- **Decision boundary visualisation** — trained on `make_moons` dataset
- **19 tests** — gradient verification via central-difference numerical differentiation
- **CI/CD** — GitHub Actions pipeline on every push
- **Live demo** — interactive Gradio app on Hugging Face Spaces

---

## Project Structure

```
ann-foundation/
├── src/
│   ├── engine.py        # Core Value class — autograd engine
│   ├── nn.py            # Neuron, Layer, MLP
│   ├── utils.py         # Dataset generation, plotting helpers
│   ├── visualize.py     # Decision boundary and training curve plots
│   └── config.py        # TrainingConfig dataclass
│
├── tests/
│   └── test_engine.py   # 19 gradient verification tests
│
├── notebooks/
│   └── Automatic-Gradient.ipynb   # Step-by-step learning notebook
│
├── examples/
│   └── Decision-Boundary-Visualization.ipynb
│
├── assets/
│   └── computation_graph.png      # Architecture diagram
│
├── app.py               # Gradio demo (Hugging Face Spaces)
├── ARCHITECTURE.md      # System design and complexity analysis
├── CONTRIBUTING.md      # Contribution guidelines
├── pyproject.toml
├── requirements.txt
└── requirements_space.txt
```

---

## Installation

```bash
git clone https://github.com/rahulkp-ai/ann-foundation.git
cd ann-foundation

# Install core package
pip install -e .

# Install with dev tools (notebooks + testing + demo)
pip install -e ".[dev]"
```

---

## Quick Start

### Autograd engine

```python
from src.engine import Value

a = Value(2.0)
b = Value(3.0)
c = a * b + a      # c = 8.0

c.backward()

print(a.grad)      # 4.0  (dc/da = b + 1 = 4)
print(b.grad)      # 2.0  (dc/db = a = 2)
```

### Build and train an MLP

```python
from src.engine import Value
from src.nn import MLP

model = MLP(nin=3, nouts=[4, 4, 1])

data = [
    ([2.0,  3.0, -1.0],  1.0),
    ([3.0, -1.0,  0.5], -1.0),
    ([0.5,  1.0,  1.0], -1.0),
    ([1.0,  1.0, -1.0],  1.0),
]

for epoch in range(50):
    total_loss = Value(0.0)
    for x, y_true in data:
        y_pred = model(x)
        loss = (y_pred - y_true) ** 2
        total_loss = total_loss + loss

    model.zero_grad()
    total_loss.backward()

    for p in model.parameters():
        p.data -= 0.05 * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | Loss: {total_loss.data:.4f}")
```

### Using TrainingConfig

```python
from src.config import TrainingConfig
from src.nn import MLP

cfg = TrainingConfig(
    hidden_layers=[8, 8],
    learning_rate=0.01,
    epochs=200,
    activation='tanh',
)

model = MLP(nin=2, nouts=cfg.hidden_layers + [1])
```

---

## Activation Functions

All three activations are implemented in `engine.py` with verified analytical gradients:

| Function  | Formula                 | Gradient            |
| --------- | ----------------------- | ------------------- |
| `tanh`    | `(e²ˣ − 1) / (e²ˣ + 1)` | `1 − tanh²(x)`      |
| `relu`    | `max(0, x)`             | `1 if x > 0 else 0` |
| `sigmoid` | `1 / (1 + e⁻ˣ)`         | `σ(x) · (1 − σ(x))` |

```python
x = Value(0.5)
print(x.tanh())     # Value(data=0.4621, grad=0.0000)
print(x.relu())     # Value(data=0.5000, grad=0.0000)
print(x.sigmoid())  # Value(data=0.6225, grad=0.0000)
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 19 tests verify analytical gradients against numerical approximation
(central difference, `h = 1e-5`). Maximum allowed error: `1e-5`.

```
tests/test_engine.py::test_add_gradient           PASSED
tests/test_engine.py::test_mul_gradient           PASSED
tests/test_engine.py::test_sigmoid_gradient       PASSED
tests/test_engine.py::test_chain_rule             PASSED
tests/test_engine.py::test_gradient_accumulation  PASSED
tests/test_engine.py::test_mlp_backward           PASSED
... (19 total)
```

With coverage:

```bash
pytest tests/ --cov=src --cov-report=term-missing
```

---

## Live Demo

An interactive Gradio demo is deployed on Hugging Face Spaces:

**[huggingface.co/spaces/rahulkp-ai/ann-foundation](https://huggingface.co/spaces/rahulkp-ai/ann-foundation)**

Two tabs:

- **Autograd Engine** — adjust input values with sliders, watch the forward pass
  compute and `backward()` accumulate gradients in real time
- **MLP Training** — configure epochs and learning rate, train on XOR data,
  see loss curve and final predictions

To run the demo locally:

```bash
python app.py
# Opens at http://localhost:7860
```

---

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full system design, including:

- `Value` class internals and the `_backward` closure pattern
- Topological sort implementation for the backward pass
- Gradient verification methodology
- MLP composition: `Neuron → Layer → MLP`
- Time and space complexity analysis

### Complexity summary

| Operation              | Time     | Space      |
| ---------------------- | -------- | ---------- |
| Forward pass (n nodes) | O(n)     | O(n)       |
| Topological sort       | O(V + E) | O(V)       |
| Backward pass          | O(n)     | O(1) extra |

### Limitations (by design)

- **Scalar only** — no tensor support. For tensors, see PyTorch.
- **No GPU** — pure Python, CPU only.
- **No batching** — each sample is a separate forward pass.

These constraints are intentional. The implementation prioritises
readability and mathematical transparency over performance.

---

## Requirements

**Core (`requirements.txt`):**

```
numpy>=1.24
matplotlib>=3.7
scikit-learn>=1.3
graphviz>=0.20
pytest>=7.0
gradio>=4.0.0
nbconvert>=7.16.0
jupyter_client>=8.0.0
nbformat>=5.10.0
ipykernel>=6.29.0
```

**Hugging Face Spaces (`requirements_space.txt`):**

```
gradio>=4.0.0
```

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup instructions, code style requirements,
and the process for adding new operations (including the gradient verification test
that every new operator must pass before merging).

---

## Learning Path

This repository sits within a structured AI/ML learning journey:

```
1. Mathematics for Computing
2. Linear Algebra
3. Artificial Neural Networks   ← this repo
4. Deep Learning Architectures
5. Generative AI Systems
```

---

## Author

**Rahul K P**
MSc Computer Science — University of Calicut (2026)

[![LinkedIn](https://img.shields.io/badge/LinkedIn-rahulkp--ai-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/rahulkp-ai)
[![GitHub](https://img.shields.io/badge/GitHub-rahulkp--ai-181717?style=flat-square&logo=github)](https://github.com/rahulkp-ai)
[![Kaggle](https://img.shields.io/badge/Kaggle-rahulkpai-20BEFF?style=flat-square&logo=kaggle)](https://kaggle.com/rahulkpai)

---

## License

MIT — see [LICENSE](LICENSE)
