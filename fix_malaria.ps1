# Script PowerShell pour résoudre les problèmes Malaria

Write-Host "=== Script de résolution des problèmes Malaria ===" -ForegroundColor Green

# Vérifier Python
Write-Host "Vérification de Python..." -ForegroundColor Yellow
python --version

# Installer les dépendances
Write-Host "Installation des dépendances..." -ForegroundColor Yellow
$packages = @("PyYAML", "Pillow", "torch", "torchvision", "dvc")

foreach ($package in $packages) {
    Write-Host "Installation de $package..." -ForegroundColor Cyan
    pip install $package
}

# Vérifier les installations
Write-Host "Test des imports..." -ForegroundColor Yellow

$testScript = @"
try:
    import yaml
    print('✓ PyYAML OK')
except ImportError:
    print('✗ PyYAML KO')

try:
    import PIL
    print('✓ Pillow OK')
except ImportError:
    print('✗ Pillow KO')

try:
    import torch
    print(f'✓ PyTorch OK: {torch.__version__}')
except ImportError:
    print('✗ PyTorch KO')

try:
    import torchvision
    print(f'✓ TorchVision OK: {torchvision.__version__}')
except ImportError:
    print('✗ TorchVision KO')

try:
    import dvc
    print('✓ DVC OK')
except ImportError:
    print('✗ DVC KO')
"@

python -c $testScript

# Vérifier la structure des données
Write-Host "Vérification des données..." -ForegroundColor Yellow

$dataTestScript = @"
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
"@

python -c $dataTestScript

# Test PyTorch ImageFolder
Write-Host "Test PyTorch ImageFolder..." -ForegroundColor Yellow

$torchTestScript = @"
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
        
        # Test d'un échantillon
        if len(train_ds) > 0:
            sample, label = train_ds[0]
            print(f'✓ Échantillon: {sample.shape}, Label: {label}')
    else:
        print('✗ Répertoire train inexistant')
        
except Exception as e:
    print(f'✗ Erreur ImageFolder: {e}')
    import traceback
    traceback.print_exc()
"@

python -c $torchTestScript

# Test DVC si disponible
Write-Host "Test DVC..." -ForegroundColor Yellow

if (Test-Path "dvc.yaml") {
    Write-Host "✓ dvc.yaml trouvé" -ForegroundColor Green
    Write-Host "Tentative d'exécution de dvc repro train..." -ForegroundColor Cyan
    
    try {
        dvc repro train
        Write-Host "✓ dvc repro train réussi!" -ForegroundColor Green
    }
    catch {
        Write-Host "✗ Erreur lors de dvc repro train: $($_.Exception.Message)" -ForegroundColor Red
        Write-Host "Tentative avec python direct..." -ForegroundColor Yellow
        
        # Essayer d'exécuter le script d'entraînement directement
        if (Test-Path "src/train.py") {
            Write-Host "Exécution directe de src/train.py..." -ForegroundColor Cyan
            python src/train.py
        }
    }
}
else {
    Write-Host "✗ dvc.yaml non trouvé" -ForegroundColor Red
}

Write-Host "=== Script terminé ===" -ForegroundColor Green
