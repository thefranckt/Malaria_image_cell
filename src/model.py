"""
Module de construction et d'entraînement du modèle de classification.

Implémente l'architecture ResNet avec transfer learning et
l'entraînement complet avec métriques et checkpointing.
"""

import json
import time
from pathlib import Path
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import Config, get_config


def build_model(config: Config = None) -> nn.Module:
    """
    Construit le modèle de classification.
    
    Args:
        config: Configuration du projet
        
    Returns:
        Modèle PyTorch initialisé
        
    Example:
        >>> from malaria_classifier import build_model, Config
        >>> config = Config.default()
        >>> model = build_model(config)
    """
    if config is None:
        config = get_config()
    
    # Chargement du modèle pré-entraîné
    if config.model.architecture.startswith("resnet"):
        weights = "DEFAULT" if config.model.pretrained else None
        model = getattr(models, config.model.architecture)(weights=weights)
        
        # Adaptation de la couche finale
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, config.model.num_classes)
    else:
        raise ValueError(
            f"Architecture non supportée: {config.model.architecture}"
        )
    
    return model


def get_data_loaders(
    config: Config = None
) -> Tuple[DataLoader, DataLoader]:
    """
    Crée les data loaders pour l'entraînement et le test.
    
    Args:
        config: Configuration du projet
        
    Returns:
        Tuple (train_loader, test_loader)
    """
    if config is None:
        config = get_config()
    
    # Transformations standards pour ImageNet
    transform = transforms.Compose([
        transforms.Resize((config.data.img_size, config.data.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Chargement des datasets
    train_dataset = datasets.ImageFolder(
        root=config.data.processed_dir / "train",
        transform=transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=config.data.processed_dir / "test",
        transform=transform
    )
    
    # Création des loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.train.batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.train.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    return train_loader, test_loader


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Entraîne le modèle sur une époque.
    
    Args:
        model: Modèle PyTorch
        loader: DataLoader d'entraînement
        criterion: Fonction de perte
        optimizer: Optimiseur
        device: Device PyTorch (cuda/cpu)
        verbose: Afficher la progression
        
    Returns:
        Tuple (loss_moyenne, accuracy)
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(loader, desc="Training", disable=not verbose)
    
    for images, labels in iterator:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
        
        # Mise à jour de la barre de progression
        if verbose:
            iterator.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
    
    return running_loss / total, correct / total


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    verbose: bool = True
) -> Tuple[float, float]:
    """
    Évalue le modèle sur un ensemble de données.
    
    Args:
        model: Modèle PyTorch
        loader: DataLoader de test
        criterion: Fonction de perte
        device: Device PyTorch
        verbose: Afficher la progression
        
    Returns:
        Tuple (loss_moyenne, accuracy)
    """
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    iterator = tqdm(loader, desc="Evaluating", disable=not verbose)
    
    with torch.no_grad():
        for images, labels in iterator:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            if verbose:
                iterator.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
    
    return running_loss / total, correct / total


def train_model(
    config: Config = None,
    verbose: bool = True
) -> Tuple[nn.Module, Dict]:
    """
    Entraîne le modèle complet et sauvegarde les résultats.
    
    Args:
        config: Configuration du projet
        verbose: Afficher les logs d'entraînement
        
    Returns:
        Tuple (modèle_entraîné, historique_métriques)
        
    Example:
        >>> from malaria_classifier import train_model
        >>> model, history = train_model()
        >>> print(f"Best accuracy: {max(history['test_acc']):.2%}")
    """
    if config is None:
        config = get_config()
    
    # Configuration du device
    if config.train.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(config.train.device)
    
    if verbose:
        print(f"🖥️  Device: {device}")
    
    # Construction du modèle
    model = build_model(config).to(device)
    train_loader, test_loader = get_data_loaders(config)
    
    # Loss et optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(
        model.parameters(),
        lr=config.train.learning_rate,
        momentum=config.train.momentum,
        weight_decay=config.train.weight_decay
    )
    
    # Historique
    history = {
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": []
    }
    
    # Boucle d'entraînement
    for epoch in range(1, config.train.epochs + 1):
        if verbose:
            print(f"\n📊 Epoch {epoch}/{config.train.epochs}")
        
        start_time = time.time()
        
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, verbose
        )
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, device, verbose
        )
        
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        
        elapsed = time.time() - start_time
        
        if verbose:
            print(f"⏱️  Temps: {elapsed:.1f}s")
            print(f"📈 Train - Loss: {train_loss:.4f}, Acc: {train_acc:.4f}")
            print(f"📉 Test  - Loss: {test_loss:.4f}, Acc: {test_acc:.4f}")
    
    # Sauvegarde du modèle
    model_path = config.paths.models_dir / "model.pth"
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'model_architecture': config.model.architecture,
        'num_classes': config.model.num_classes,
        'class_names': list(config.model.class_names),
        'img_size': config.data.img_size,
        'config': {
            'batch_size': config.train.batch_size,
            'learning_rate': config.train.learning_rate,
            'epochs': config.train.epochs
        },
        'final_accuracy': history["test_acc"][-1],
        'best_accuracy': max(history["test_acc"])
    }
    
    torch.save(checkpoint, model_path)
    
    # Sauvegarde des métriques
    metrics = {
        'training_history': history,
        'final_metrics': {
            'test_accuracy': history["test_acc"][-1],
            'test_loss': history["test_loss"][-1],
            'best_test_accuracy': max(history["test_acc"]),
            'total_epochs': config.train.epochs
        },
        'model_info': {
            'architecture': config.model.architecture,
            'pretrained': config.model.pretrained,
            'num_classes': config.model.num_classes,
            'class_names': list(config.model.class_names)
        }
    }
    
    with open(config.paths.metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    if verbose:
        print(f"\n✅ Entraînement terminé!")
        print(f"📁 Modèle sauvegardé: {model_path}")
        print(f"📊 Métriques sauvegardées: {config.paths.metrics_file}")
        print(f"🎯 Meilleure accuracy: {max(history['test_acc']):.4f}")
    
    return model, history


if __name__ == "__main__":
    # Point d'entrée pour exécution standalone
    train_model(verbose=True)
