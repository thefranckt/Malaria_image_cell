import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import yaml
import json
from pathlib import Path
import numpy as np

class MalariaClassifier:
    def __init__(self, model_path="models/model.pth", params_path="params.yaml"):
        """
        Classificateur de malaria pour le déploiement
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.classes = ["Parasitized", "Uninfected"]
        
        # Charger les paramètres
        with open(params_path) as f:
            self.params = yaml.safe_load(f)
        
        # Initialiser le modèle
        self.model = self._load_model(model_path)
        
        # Définir les transformations
        self.transform = transforms.Compose([
            transforms.Resize((self.params["preprocess"]["img_size"], 
                             self.params["preprocess"]["img_size"])),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                               [0.229, 0.224, 0.225])
        ])
    
    def _load_model(self, model_path):
        """Charger le modèle pré-entraîné"""
        # Créer l'architecture (utiliser weights=None au lieu de pretrained=False)
        model = getattr(models, self.params["model"]["architecture"])(weights=None)
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(self.classes))
        
        # Charger les poids depuis le checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        # Si c'est un dictionnaire avec metadata, extraire le state_dict
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            # Compatibilité avec ancien format
            model.load_state_dict(checkpoint)
        
        model.eval()
        model.to(self.device)
        
        return model
    
    def predict_single(self, image_path):
        """
        Prédiction sur une seule image
        
        Args:
            image_path: chemin vers l'image
            
        Returns:
            dict: {'class': str, 'confidence': float, 'probabilities': dict}
        """
        # Charger et préprocesser l'image
        image = Image.open(image_path).convert('RGB')
        input_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        # Prédiction
        with torch.no_grad():
            outputs = self.model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        predicted_class = self.classes[predicted.item()]
        confidence_score = confidence.item()
        
        # Probabilités pour chaque classe
        prob_dict = {
            self.classes[i]: probabilities[0][i].item() 
            for i in range(len(self.classes))
        }
        
        return {
            'class': predicted_class,
            'confidence': confidence_score,
            'probabilities': prob_dict
        }
    
    def predict_batch(self, image_paths):
        """
        Prédiction sur un batch d'images
        
        Args:
            image_paths: liste des chemins d'images
            
        Returns:
            list: liste des résultats de prédiction
        """
        results = []
        for image_path in image_paths:
            try:
                result = self.predict_single(image_path)
                result['image_path'] = str(image_path)
                results.append(result)
            except Exception as e:
                results.append({
                    'image_path': str(image_path),
                    'error': str(e)
                })
        
        return results
    
    def save_predictions(self, results, output_path):
        """Sauvegarder les résultats"""
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

def main():
    """Exemple d'utilisation"""
    classifier = MalariaClassifier()
    
    # Test sur une image
    test_image = "data/processed/test/Parasitized/C100P61ThinF_IMG_20150918_144104_cell_162.png"
    if Path(test_image).exists():
        result = classifier.predict_single(test_image)
        print(f"Prédiction: {result['class']} (confiance: {result['confidence']:.3f})")
        print(f"Probabilités: {result['probabilities']}")

if __name__ == "__main__":
    main()
