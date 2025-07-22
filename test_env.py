#!/usr/bin/env python3
"""
Script de test pour vérifier l'environnement et diagnostiquer les problèmes.
"""

import sys
import os
from pathlib import Path

def test_environment():
    print("=== Test de l'environnement ===")
    print(f"Python version: {sys.version}")
    print(f"Python executable: {sys.executable}")
    print(f"Working directory: {os.getcwd()}")
    print()
    
    print("=== Test des imports ===")
    try:
        import yaml
        print("✓ PyYAML disponible")
    except ImportError as e:
        print(f"✗ PyYAML non disponible: {e}")
    
    try:
        from PIL import Image
        print("✓ Pillow disponible")
    except ImportError as e:
        print(f"✗ Pillow non disponible: {e}")
    
    try:
        import torch
        print(f"✓ PyTorch disponible: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch non disponible: {e}")
    
    print()
    
    print("=== Test de la structure des fichiers ===")
    params_file = Path("params.yaml")
    if params_file.exists():
        print("✓ params.yaml trouvé")
        try:
            with open(params_file, 'r') as f:
                import yaml
                params = yaml.safe_load(f)
                print(f"  - raw_dir: {params['data']['raw_dir']}")
                print(f"  - processed_dir: {params['data']['processed_dir']}")
        except Exception as e:
            print(f"✗ Erreur lors de la lecture de params.yaml: {e}")
    else:
        print("✗ params.yaml non trouvé")
    
    # Vérifier les répertoires de données
    data_dir = Path("data")
    if data_dir.exists():
        print("✓ Répertoire data/ trouvé")
        raw_dir = data_dir / "raw" / "cell_images"
        if raw_dir.exists():
            parasitized = raw_dir / "Parasitized"
            uninfected = raw_dir / "Uninfected"
            print(f"  - Parasitized: {len(list(parasitized.glob('*.png')))} images" if parasitized.exists() else "  - Parasitized: non trouvé")
            print(f"  - Uninfected: {len(list(uninfected.glob('*.png')))} images" if uninfected.exists() else "  - Uninfected: non trouvé")
        else:
            print("✗ Répertoire raw/cell_images non trouvé")
        
        processed_dir = data_dir / "processed"
        if processed_dir.exists():
            print("✓ Répertoire processed/ trouvé")
            train_dir = processed_dir / "train"
            test_dir = processed_dir / "test"
            if train_dir.exists():
                train_para = train_dir / "Parasitized"
                train_uninf = train_dir / "Uninfected"
                print(f"  - Train Parasitized: {len(list(train_para.glob('*.png')))} images" if train_para.exists() else "  - Train Parasitized: non trouvé")
                print(f"  - Train Uninfected: {len(list(train_uninf.glob('*.png')))} images" if train_uninf.exists() else "  - Train Uninfected: non trouvé")
            if test_dir.exists():
                test_para = test_dir / "Parasitized"
                test_uninf = test_dir / "Uninfected"
                print(f"  - Test Parasitized: {len(list(test_para.glob('*.png')))} images" if test_para.exists() else "  - Test Parasitized: non trouvé")
                print(f"  - Test Uninfected: {len(list(test_uninf.glob('*.png')))} images" if test_uninf.exists() else "  - Test Uninfected: non trouvé")
        else:
            print("✗ Répertoire processed/ non trouvé")
    else:
        print("✗ Répertoire data/ non trouvé")

if __name__ == "__main__":
    test_environment()
