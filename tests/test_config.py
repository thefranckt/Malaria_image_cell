"""
Tests pour le module de configuration.
"""

import pytest
from pathlib import Path
from src.config import Config, DataConfig, ModelConfig, TrainConfig


def test_data_config_defaults():
    """Test les valeurs par défaut de DataConfig."""
    config = DataConfig()
    assert config.img_size == 64
    assert config.test_split == 0.2
    assert config.random_seed == 42


def test_data_config_validation():
    """Test la validation des paramètres."""
    with pytest.raises(AssertionError):
        DataConfig(test_split=1.5)  # > 1
    
    with pytest.raises(AssertionError):
        DataConfig(img_size=-10)  # négatif


def test_model_config_defaults():
    """Test les valeurs par défaut de ModelConfig."""
    config = ModelConfig()
    assert config.architecture == "resnet18"
    assert config.pretrained == True
    assert config.num_classes == 2
    assert len(config.class_names) == 2


def test_train_config_defaults():
    """Test les valeurs par défaut de TrainConfig."""
    config = TrainConfig()
    assert config.batch_size == 32
    assert config.epochs == 10
    assert config.learning_rate == 0.001


def test_config_creation():
    """Test la création de la config complète."""
    config = Config.default()
    assert config.data.img_size == 64
    assert config.model.architecture == "resnet18"
    assert config.train.batch_size == 32
