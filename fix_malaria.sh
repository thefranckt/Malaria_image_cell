#!/bin/bash

echo "=== Script de résolution des problèmes Malaria ==="

# Vérifier Python
echo "Python version:"
python --version

# Installer les dépendances
echo "Installation des dépendances..."
pip install PyYAML Pillow torch torchvision dvc

# Vérifier l'installation
echo "Test des imports..."
python -c "
try:
    import yaml; print('✓ PyYAML OK')
except: print('✗ PyYAML KO')

try:
    import PIL; print('✓ Pillow OK') 
except: print('✗ Pillow KO')

try:
    import torch; print(f'✓ PyTorch OK: {torch.__version__}')
except: print('✗ PyTorch KO')

try:
    import torchvision; print(f'✓ TorchVision OK: {torchvision.__version__}')
except: print('✗ TorchVision KO')
"

# Vérifier la structure des données
echo "Vérification des données..."
python -c "
from pathlib import Path

paths = [
    'data/processed/train/Parasitized',
    'data/processed/train/Uninfected',
    'data/processed/test/Parasitized', 
    'data/processed/test/Uninfected'
]

for path_str in paths:
    path = Path(path_str)
    if path.exists():
        count = len(list(path.glob('*.png')))
        print(f'✓ {path_str}: {count} images')
    else:
        print(f'✗ {path_str}: n\'existe pas')
"

# Test PyTorch ImageFolder
echo "Test PyTorch ImageFolder..."
python -c "
try:
    import torch
    from torchvision import datasets, transforms
    from pathlib import Path
    
    transform = transforms.Compose([transforms.ToTensor()])
    train_dir = Path('data/processed/train')
    
    if train_dir.exists():
        train_ds = datasets.ImageFolder(train_dir, transform=transform)
        print(f'✓ ImageFolder OK: {len(train_ds)} échantillons')
        print(f'✓ Classes: {train_ds.classes}')
    else:
        print('✗ Répertoire train inexistant')
        
except Exception as e:
    print(f'✗ Erreur ImageFolder: {e}')
"

# Tester l'entraînement DVC
echo "Test DVC..."
if [ -f "dvc.yaml" ]; then
    echo "✓ dvc.yaml trouvé"
    echo "Exécution de dvc repro train..."
    dvc repro train
else
    echo "✗ dvc.yaml non trouvé"
fi

echo "=== Script terminé ==="
