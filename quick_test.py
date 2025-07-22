"""
Script de test rapide pour l'entraînement avec un mini-modèle
"""

import os
from pathlib import Path

def quick_test():
    print("=== Test rapide ===")
    
    # 1. Vérifier que les données existent
    train_para = Path("data/processed/train/Parasitized")
    train_uninf = Path("data/processed/train/Uninfected")
    
    if train_para.exists() and train_uninf.exists():
        para_count = len(list(train_para.glob("*.png")))
        uninf_count = len(list(train_uninf.glob("*.png")))
        print(f"✓ Données d'entraînement trouvées:")
        print(f"  - Parasitized: {para_count}")
        print(f"  - Uninfected: {uninf_count}")
    else:
        print("✗ Données d'entraînement manquantes")
        return False
    
    # 2. Test d'import PyTorch
    try:
        import torch
        import torchvision
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        print(f"✓ PyTorch {torch.__version__} importé")
    except ImportError as e:
        print(f"✗ Erreur d'import PyTorch: {e}")
        print("Exécutez d'abord: pip install torch torchvision")
        return False
    
    # 3. Test ImageFolder
    try:
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor()
        ])
        
        train_dataset = datasets.ImageFolder(
            "data/processed/train",
            transform=transform
        )
        
        print(f"✓ Dataset créé avec {len(train_dataset)} échantillons")
        print(f"✓ Classes: {train_dataset.classes}")
        
        # Test DataLoader
        train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
        
        # Test d'un batch
        batch = next(iter(train_loader))
        images, labels = batch
        print(f"✓ Batch testé: {images.shape}, labels: {labels}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test ImageFolder: {e}")
        return False

def run_mini_training():
    """Lance un entraînement minimal pour tester"""
    
    print("\n=== Test d'entraînement minimal ===")
    
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torchvision import datasets, transforms
        from torch.utils.data import DataLoader
        
        # Configuration
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: {device}")
        
        # Transforms
        transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
            transforms.Normalize([0.485], [0.456])  # Grayscale normalization
        ])
        
        # Dataset
        train_dataset = datasets.ImageFolder("data/processed/train", transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
        
        # Modèle simple
        class SimpleCNN(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(3, 16, 3),  # 3 canaux pour RGB
                    nn.ReLU(),
                    nn.AdaptiveAvgPool2d((8, 8)),
                    nn.Flatten(),
                    nn.Linear(16 * 8 * 8, 64),
                    nn.ReLU(),
                    nn.Linear(64, 2)  # 2 classes
                )
            
            def forward(self, x):
                return self.features(x)
        
        model = SimpleCNN().to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        
        print("✓ Modèle créé")
        
        # Test d'une époque
        model.train()
        total_loss = 0
        num_batches = 0
        
        for i, (images, labels) in enumerate(train_loader):
            if i >= 5:  # Seulement 5 batches pour le test
                break
                
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            print(f"Batch {i+1}: Loss = {loss.item():.4f}")
        
        avg_loss = total_loss / num_batches
        print(f"✓ Test d'entraînement réussi! Perte moyenne: {avg_loss:.4f}")
        
        return True
        
    except Exception as e:
        print(f"✗ Erreur lors du test d'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_test()
    
    if success:
        run_mini_training()
    else:
        print("Veuillez d'abord résoudre les problèmes ci-dessus.")
