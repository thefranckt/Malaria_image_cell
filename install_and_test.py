"""
Script d'installation et de diagnostic pour le projet Malaria
"""

import subprocess
import sys
import os
from pathlib import Path

def install_package(package):
    """Installe un package avec pip"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installé avec succès")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Erreur lors de l'installation de {package}: {e}")
        return False

def check_and_install_dependencies():
    """Vérifie et installe les dépendances nécessaires"""
    
    dependencies = [
        "PyYAML",
        "Pillow", 
        "torch",
        "torchvision",
        "dvc"
    ]
    
    print("=== Installation des dépendances ===")
    
    for dep in dependencies:
        print(f"Installation de {dep}...")
        install_package(dep)
    
    print("\n=== Vérification des installations ===")
    
    # Test imports
    try:
        import yaml
        print("✓ PyYAML disponible")
    except ImportError:
        print("✗ PyYAML non disponible")
    
    try:
        from PIL import Image
        print("✓ Pillow disponible")
    except ImportError:
        print("✗ Pillow non disponible")
    
    try:
        import torch
        import torchvision
        print(f"✓ PyTorch disponible: {torch.__version__}")
        print(f"✓ TorchVision disponible: {torchvision.__version__}")
    except ImportError:
        print("✗ PyTorch/TorchVision non disponible")
    
    try:
        import dvc
        print(f"✓ DVC disponible")
    except ImportError:
        print("✗ DVC non disponible")

def check_data_structure():
    """Vérifie la structure des données"""
    
    print("\n=== Vérification de la structure des données ===")
    
    # Vérifier les répertoires
    paths_to_check = [
        "data/processed/train/Parasitized",
        "data/processed/train/Uninfected", 
        "data/processed/test/Parasitized",
        "data/processed/test/Uninfected"
    ]
    
    for path_str in paths_to_check:
        path = Path(path_str)
        if path.exists():
            count = len(list(path.glob("*.png")))
            print(f"✓ {path_str}: {count} images")
        else:
            print(f"✗ {path_str}: n'existe pas")

def test_pytorch_imagefolder():
    """Test PyTorch ImageFolder avec nos données"""
    
    print("\n=== Test PyTorch ImageFolder ===")
    
    try:
        import torch
        from torchvision import datasets, transforms
        from pathlib import Path
        
        # Transformer simple pour le test
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        
        # Test sur le répertoire train
        train_dir = Path("data/processed/train")
        if train_dir.exists():
            try:
                train_ds = datasets.ImageFolder(train_dir, transform=transform)
                print(f"✓ ImageFolder fonctionne sur train/")
                print(f"  - Nombre d'échantillons: {len(train_ds)}")
                print(f"  - Classes: {train_ds.classes}")
                print(f"  - Nombre de classes: {len(train_ds.classes)}")
                
                # Test de chargement d'un échantillon
                if len(train_ds) > 0:
                    sample, label = train_ds[0]
                    print(f"  - Forme d'un échantillon: {sample.shape}")
                    print(f"  - Label d'exemple: {label} ({train_ds.classes[label]})")
            except Exception as e:
                print(f"✗ Erreur avec ImageFolder sur train/: {e}")
        else:
            print("✗ Répertoire train/ n'existe pas")
            
    except ImportError as e:
        print(f"✗ Import PyTorch échoué: {e}")

if __name__ == "__main__":
    print("=== Script de diagnostic et installation ===")
    print(f"Python: {sys.version}")
    print(f"Répertoire: {os.getcwd()}")
    
    # Installation
    check_and_install_dependencies()
    
    # Vérification des données
    check_data_structure()
    
    # Test PyTorch
    test_pytorch_imagefolder()
    
    print("\n=== Diagnostic terminé ===")
