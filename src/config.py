"""
Configuration centralisée pour le projet Malaria Cell Classification.

Ce module gère tous les paramètres du projet de manière Pythonic,
avec validation et documentation complète.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional
import yaml


@dataclass
class DataConfig:
    """Configuration des données."""
    raw_dir: Path = Path("data/raw/cell_images")
    processed_dir: Path = Path("data/processed")
    img_size: int = 64
    test_split: float = 0.2
    random_seed: int = 42
    
    def __post_init__(self):
        """Validation des paramètres."""
        assert 0 < self.test_split < 1, "test_split doit être entre 0 et 1"
        assert self.img_size > 0, "img_size doit être positif"


@dataclass
class ModelConfig:
    """Configuration du modèle."""
    architecture: str = "resnet18"
    pretrained: bool = True
    num_classes: int = 2
    class_names: Tuple[str, ...] = ("Parasitized", "Uninfected")
    
    def __post_init__(self):
        """Validation des paramètres."""
        assert self.architecture in ["resnet18", "resnet34", "resnet50"], \
            f"Architecture {self.architecture} non supportée"
        assert self.num_classes == len(self.class_names), \
            "num_classes doit correspondre au nombre de class_names"


@dataclass
class TrainConfig:
    """Configuration de l'entraînement."""
    batch_size: int = 32
    epochs: int = 10
    learning_rate: float = 0.001
    momentum: float = 0.9
    weight_decay: float = 0.0001
    device: str = "auto"  # "auto", "cuda", "cpu"
    
    def __post_init__(self):
        """Validation des paramètres."""
        assert self.batch_size > 0, "batch_size doit être positif"
        assert self.epochs > 0, "epochs doit être positif"
        assert self.learning_rate > 0, "learning_rate doit être positif"


@dataclass
class PathsConfig:
    """Configuration des chemins de sortie."""
    models_dir: Path = Path("models")
    metrics_file: Path = Path("metrics.json")
    logs_dir: Path = Path("logs")
    
    def __post_init__(self):
        """Créer les répertoires s'ils n'existent pas."""
        self.models_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class Config:
    """Configuration principale du projet."""
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    
    @classmethod
    def from_yaml(cls, path: str = "params.yaml") -> "Config":
        """
        Charger la configuration depuis un fichier YAML.
        
        Args:
            path: Chemin vers le fichier YAML
            
        Returns:
            Instance de Config
        """
        with open(path) as f:
            params = yaml.safe_load(f)
        
        return cls(
            data=DataConfig(
                raw_dir=Path(params["data"]["raw_dir"]),
                processed_dir=Path(params["data"]["processed_dir"]),
                img_size=params["preprocess"]["img_size"],
                test_split=params["preprocess"]["test_split"],
                random_seed=params["preprocess"]["random_seed"]
            ),
            model=ModelConfig(
                architecture=params["model"]["architecture"],
                pretrained=params["model"]["pretrained"]
            ),
            train=TrainConfig(
                batch_size=params["train"]["batch_size"],
                epochs=params["train"]["epochs"],
                learning_rate=params["train"]["learning_rate"],
                momentum=params["train"]["momentum"],
                weight_decay=params["train"]["weight_decay"]
            ),
            paths=PathsConfig(
                metrics_file=Path(params.get("dvc", {}).get("metrics_file", "metrics.json"))
            )
        )
    
    @classmethod
    def default(cls) -> "Config":
        """
        Retourner la configuration par défaut.
        
        Returns:
            Instance de Config avec valeurs par défaut
        """
        return cls()


# Instance singleton pour accès global
_config: Optional[Config] = None


def get_config(reload: bool = False) -> Config:
    """
    Obtenir la configuration du projet.
    
    Args:
        reload: Si True, recharge la configuration depuis le fichier
        
    Returns:
        Instance de Config
    """
    global _config
    
    if _config is None or reload:
        try:
            _config = Config.from_yaml()
        except FileNotFoundError:
            print("⚠️  params.yaml non trouvé, utilisation de la config par défaut")
            _config = Config.default()
    
    return _config
