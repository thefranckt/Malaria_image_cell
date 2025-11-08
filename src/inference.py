"""
Module d'inférence pour la classification de malaria.

Fournit une interface simple pour charger un modèle entraîné
et effectuer des prédictions sur de nouvelles images.
"""

from pathlib import Path
from typing import Dict, List, Union

import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image

from .config import Config, get_config


class MalariaClassifier:
    """
    Classificateur de malaria pour le déploiement en production.
    
    Cette classe encapsule le modèle entraîné et fournit
    une API simple pour les prédictions.
    
    Attributes:
        model: Modèle PyTorch chargé
        device: Device utilisé (cuda/cpu)
        class_names: Liste des noms de classes
        transform: Pipeline de transformation des images
        
    Example:
        >>> from malaria_classifier import MalariaClassifier
        >>> classifier = MalariaClassifier()
        >>> result = classifier.predict("cell_image.png")
        >>> print(f"{result['class']}: {result['confidence']:.2%}")
    """
    
    def __init__(
        self,
        model_path: Union[str, Path] = "models/model.pth",
        config: Config = None
    ):
        """
        Initialise le classificateur.
        
        Args:
            model_path: Chemin vers le fichier du modèle
            config: Configuration du projet
        """
        if config is None:
            config = get_config()
        
        self.config = config
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.class_names = list(config.model.class_names)
        
        # Charger le modèle
        self.model = self._load_model(Path(model_path))
        
        # Pipeline de transformation
        self.transform = transforms.Compose([
            transforms.Resize((config.data.img_size, config.data.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
    
    def _load_model(self, model_path: Path) -> nn.Module:
        """
        Charge le modèle depuis le checkpoint.
        
        Args:
            model_path: Chemin vers le fichier du modèle
            
        Returns:
            Modèle PyTorch chargé et en mode évaluation
        """
        # Créer l'architecture
        model = getattr(models, self.config.model.architecture)(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, len(self.class_names))
        
        # Charger les poids
        checkpoint = torch.load(model_path, map_location=self.device)
        
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Compatibilité avec ancien format
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def predict(
        self,
        image_path: Union[str, Path, Image.Image]
    ) -> Dict[str, Union[str, float, Dict[str, float]]]:
        """
        Effectue une prédiction sur une image.
        
        Args:
            image_path: Chemin vers l'image ou objet PIL Image
            
        Returns:
            Dictionnaire contenant:
                - class: Classe prédite
                - confidence: Score de confiance
                - probabilities: Probabilités pour chaque classe
                
        Example:
            >>> classifier = MalariaClassifier()
            >>> result = classifier.predict("cell.png")
            >>> print(result)
            {'class': 'Parasitized', 'confidence': 0.98, 
             'probabilities': {'Parasitized': 0.98, 'Uninfected': 0.02}}
        """
        # Charger l'image
        if isinstance(image_path, (str, Path)):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path.convert('RGB')
        
        # Prétraitement
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.class_names[predicted.item()]
        confidence_score = confidence.item()
        
        # Probabilités pour toutes les classes
        prob_dict = {
            self.class_names[i]: probabilities[0][i].item()
            for i in range(len(self.class_names))
        }
        
        return {
            'class': predicted_class,
            'confidence': confidence_score,
            'probabilities': prob_dict
        }
    
    def predict_batch(
        self,
        image_paths: List[Union[str, Path]]
    ) -> List[Dict]:
        """
        Effectue des prédictions sur un batch d'images.
        
        Args:
            image_paths: Liste de chemins vers les images
            
        Returns:
            Liste de dictionnaires de résultats
            
        Example:
            >>> classifier = MalariaClassifier()
            >>> results = classifier.predict_batch(["img1.png", "img2.png"])
            >>> for r in results:
            ...     print(f"{r['image']}: {r['class']}")
        """
        results = []
        
        for image_path in image_paths:
            try:
                result = self.predict(image_path)
                result['image'] = str(image_path)
                result['success'] = True
                results.append(result)
            except Exception as e:
                results.append({
                    'image': str(image_path),
                    'success': False,
                    'error': str(e)
                })
        
        return results
    
    def __repr__(self) -> str:
        """Représentation string du classificateur."""
        return (
            f"MalariaClassifier("
            f"model={self.config.model.architecture}, "
            f"device={self.device}, "
            f"classes={self.class_names})"
        )


def predict_image(image_path: str, model_path: str = "models/model.pth") -> Dict:
    """
    Fonction utilitaire pour une prédiction rapide.
    
    Args:
        image_path: Chemin vers l'image
        model_path: Chemin vers le modèle
        
    Returns:
        Résultat de la prédiction
        
    Example:
        >>> from malaria_classifier.inference import predict_image
        >>> result = predict_image("cell.png")
        >>> print(result['class'])
    """
    classifier = MalariaClassifier(model_path=model_path)
    return classifier.predict(image_path)


if __name__ == "__main__":
    # Point d'entrée pour test rapide
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        result = predict_image(image_path)
        print(f"\n🔬 Résultat de l'analyse:")
        print(f"   Classe: {result['class']}")
        print(f"   Confiance: {result['confidence']:.2%}")
        print(f"\n📊 Probabilités:")
        for cls, prob in result['probabilities'].items():
            print(f"   {cls}: {prob:.2%}")
    else:
        print("Usage: python -m src.inference <image_path>")
