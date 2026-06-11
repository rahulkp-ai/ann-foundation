import pytest
from src.config import TrainingConfig

def test_default_config():
    cfg = TrainingConfig()
    assert cfg.learning_rate == 0.05
    assert cfg.epochs == 100
    assert cfg.activation == 'tanh'
    assert cfg.hidden_layers == [4, 4]

def test_custom_config():
    cfg = TrainingConfig(hidden_layers=[8, 4], learning_rate=0.01, epochs=50)
    assert cfg.learning_rate == 0.01
    assert cfg.epochs == 50

def test_invalid_activation_raises():
    with pytest.raises(AssertionError):
        TrainingConfig(activation='gelu')

def test_invalid_lr_raises():
    with pytest.raises(AssertionError):
        TrainingConfig(learning_rate=-0.1)

def test_invalid_epochs_raises():
    with pytest.raises(AssertionError):
        TrainingConfig(epochs=0)