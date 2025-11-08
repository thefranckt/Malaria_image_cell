"""
Malaria Cell Classification Package

Un package Python professionnel pour la classification automatique
de cellules sanguines infectées par la malaria en utilisant deep learning.

Modules:
    config: Configuration centralisée du projet
    data: Pipeline de préparation des données
    model: Architecture et entraînement du modèle
    inference: Prédictions et déploiement
    utils: Fonctions utilitaires

Example:
    >>> from malaria_classifier import Config, train_model
    >>> config = Config.from_yaml()
    >>> model = train_model(config)
"""

__version__ = "1.0.0"
__author__ = "Franckt"
__email__ = "your.email@example.com"

from .config import Config, get_config
from .data import prepare_data
from .model import build_model, train_model
from .inference import MalariaClassifier

__all__ = [
    "Config",
    "get_config",
    "prepare_data",
    "build_model",
    "train_model",
    "MalariaClassifier",
]
