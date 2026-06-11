"""
Configuration dataclass for MLP training.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class TrainingConfig:
    """
    Configuration for MLP training runs.
    
    Attributes:
        hidden_layers: List of hidden layer sizes
        learning_rate: Gradient descent step size
        epochs: Number of training iterations
        activation: Activation function ('tanh', 'relu', 'sigmoid')
        seed: Random seed for reproducibility
    """
    hidden_layers: List[int] = field(default_factory=lambda: [4, 4])
    learning_rate: float = 0.05
    epochs: int = 100
    activation: str = 'tanh'
    seed: int = 42

    def __post_init__(self):
        assert self.learning_rate > 0, "Learning rate must be positive"
        assert self.epochs > 0, "Epochs must be positive"
        assert self.activation in ('tanh', 'relu', 'sigmoid'), \
            f"Unknown activation: {self.activation}"