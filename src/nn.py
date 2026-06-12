# src/nn.py
import math
import random
from .engine import Value


class Neuron:
    def __init__(self, nin, activation='tanh'):
        # Xavier/Glorot initialization — prevents vanishing gradients at init
        # He init for ReLU (scale by sqrt(2/n)), Xavier for tanh/sigmoid (sqrt(1/n))
        scale = math.sqrt(2.0 / nin) if activation == 'relu' else math.sqrt(1.0 / nin)
        self.w = [Value(random.uniform(-scale, scale)) for _ in range(nin)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x):
        # Ensure inputs are treated as Value instances
        x = [xi if isinstance(xi, Value) else Value(xi) for xi in x]
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)

        if self.activation == 'tanh':
            return act.tanh()
        elif self.activation == 'relu':
            return act.relu()
        elif self.activation == 'sigmoid':
            return act.sigmoid()
        elif self.activation == 'linear':
            return act
        else:
            raise ValueError(f"Unknown activation: {self.activation}")

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"Neuron(nin={len(self.w)}, activation={self.activation})"


class Layer:
    def __init__(self, nin, nout, activation='tanh'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        # Ensure input x is always processed as a list/iterable
        if not isinstance(x, (list, tuple)):
            x = [x]
        outs = [n(x) for n in self.neurons]
        return outs[0] if len(outs) == 1 else outs

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer(neurons={len(self.neurons)})"


class MLP:
    def __init__(self, nin, nouts, activation='tanh'):
        sizes = [nin] + nouts
        self.layers = []

        # Hidden layers use the chosen activation
        for i in range(len(nouts) - 1):
            self.layers.append(Layer(sizes[i], sizes[i + 1], activation))

        # Output layer is always linear so MSE loss computes cleanly
        self.layers.append(Layer(sizes[-2], sizes[-1], activation='linear'))

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0.0

    def __repr__(self):
        return f"MLP(layers={self.layers})"