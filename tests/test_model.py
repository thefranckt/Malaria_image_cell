"""
Tests pour le module de modèle.
"""

import pytest
import torch
from src.config import Config
from src.model import build_model


def test_build_model_default():
    """Test la construction du modèle avec config par défaut."""
    config = Config.default()
    model = build_model(config)
    
    assert model is not None
    assert hasattr(model, 'fc')
    assert model.fc.out_features == 2


def test_model_forward_pass():
    """Test un forward pass simple."""
    config = Config.default()
    model = build_model(config)
    model.eval()
    
    # Image factice 64x64
    dummy_input = torch.randn(1, 3, 64, 64)
    
    with torch.no_grad():
        output = model(dummy_input)
    
    assert output.shape == (1, 2)


def test_unsupported_architecture():
    """Test qu'une architecture non supportée lève une erreur."""
    config = Config.default()
    config.model.architecture = "unsupported_model"
    
    with pytest.raises(ValueError):
        build_model(config)
