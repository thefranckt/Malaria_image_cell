#!/usr/bin/env python3

# Test simple pour diagnostiquer le problème
print("=== TEST DIAGNOSTIC ===")

try:
    import os
    print("✓ os importé")
    
    import yaml
    print("✓ yaml importé")
    
    import torch
    print(f"✓ torch importé - version: {torch.__version__}")
    
    import torchvision
    print(f"✓ torchvision importé - version: {torchvision.__version__}")
    
    from torchvision import datasets, transforms, models
    print("✓ modules torchvision importés")
    
    # Test de chargement des paramètres
    if os.path.exists("params.yaml"):
        with open("params.yaml") as f:
            params = yaml.safe_load(f)
        print("✓ params.yaml chargé")
        print(f"  - Architecture: {params.get('model', {}).get('architecture', 'Non défini')}")
    else:
        print("✗ params.yaml non trouvé")
        
    # Test d'existence des données processées
    processed_dir = "data/processed"
    if os.path.exists(processed_dir):
        train_dir = os.path.join(processed_dir, "train")
        test_dir = os.path.join(processed_dir, "test")
        
        if os.path.exists(train_dir):
            classes = os.listdir(train_dir)
            print(f"✓ Dossier train trouvé avec classes: {classes}")
            
            for class_name in classes:
                class_path = os.path.join(train_dir, class_name)
                if os.path.isdir(class_path):
                    count = len([f for f in os.listdir(class_path) if f.endswith('.png')])
                    print(f"  - {class_name}: {count} images")
        else:
            print("✗ Dossier train non trouvé")
            
        if os.path.exists(test_dir):
            classes = os.listdir(test_dir)
            print(f"✓ Dossier test trouvé avec classes: {classes}")
        else:
            print("✗ Dossier test non trouvé")
    else:
        print("✗ Dossier processed non trouvé")
        
    print("\n=== TEST COMPLET RÉUSSI ===")
    
except Exception as e:
    print(f"✗ ERREUR: {e}")
    import traceback
    traceback.print_exc()
