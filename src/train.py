import os
import yaml
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from pathlib import Path

def load_params(path="params.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)

def save_metrics(metrics, out_path):
    with open(out_path, "w") as f:
        json.dump(metrics, f, indent=2)

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    return running_loss/total, correct/total

def eval_one_epoch(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return running_loss/total, correct/total

def main():
    # Charger les paramètres
    params = load_params()
    p_data = params["data"]
    p_p = params["preprocess"]
    p_t = params["train"]
    p_m = params["model"]
    dvc = params["dvc"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Transforms
    transform = transforms.Compose([
        transforms.Resize((p_p["img_size"], p_p["img_size"])),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # Datasets & DataLoaders
    train_dir = Path(p_data["processed_dir"]) / "train"
    test_dir  = Path(p_data["processed_dir"]) / "test"

    train_ds = datasets.ImageFolder(train_dir, transform=transform)
    test_ds  = datasets.ImageFolder(test_dir,  transform=transform)

    train_loader = DataLoader(train_ds,
                              batch_size=p_t["batch_size"],
                              shuffle=True)
    test_loader  = DataLoader(test_ds,
                              batch_size=p_t["batch_size"],
                              shuffle=False)

    # Modèle
    if p_m["architecture"].startswith("resnet"):
        # Utiliser weights au lieu de pretrained (nouvelle API)
        if p_m["pretrained"]:
            weights = "DEFAULT"  # Utilise les poids par défaut (ImageNet)
        else:
            weights = None
        model = getattr(models, p_m["architecture"])(weights=weights)
        # Ajuster la tête
        in_feats = model.fc.in_features
        model.fc = nn.Linear(in_feats, len(train_ds.classes))
    else:
        raise ValueError("Architecture non supportée")

    model = model.to(device)

    # Critère & Optimiseur
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(),
                          lr=p_t["learning_rate"],
                          momentum=p_t["momentum"],
                          weight_decay=p_t["weight_decay"])

    # Boucle d’entraînement
    history = {"train_loss":[], "train_acc":[], "test_loss":[], "test_acc":[]}
    for epoch in range(1, p_t["epochs"]+1):
        start = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader,
                                                criterion, optimizer, device)
        test_loss,  test_acc  = eval_one_epoch(model, test_loader,
                                               criterion, device)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        print(f"Epoch {epoch}/{p_t['epochs']} "
              f"– train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
              f"– test_loss={test_loss:.4f} test_acc={test_acc:.4f} "
              f"({time.time()-start:.1f}s)")

    # Sauvegarde du modèle avec métadonnées
    out_dir = Path("models")
    out_dir.mkdir(exist_ok=True)
    model_path = out_dir / "model.pth"
    
    # Sauvegarder le modèle avec les métadonnées
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_architecture': p_m["architecture"],
        'num_classes': len(train_ds.classes),
        'class_names': train_ds.classes,
        'img_size': p_p["img_size"],
        'train_params': p_t,
        'final_accuracy': history["test_acc"][-1]
    }, model_path)

    # Sauvegarde des métriques avec informations additionnelles
    final_metrics = {
        'training_history': history,
        'final_test_accuracy': history["test_acc"][-1],
        'final_test_loss': history["test_loss"][-1],
        'best_test_accuracy': max(history["test_acc"]),
        'total_epochs': p_t["epochs"],
        'model_info': {
            'architecture': p_m["architecture"],
            'pretrained': p_m["pretrained"],
            'num_classes': len(train_ds.classes),
            'class_names': train_ds.classes
        },
        'dataset_info': {
            'train_samples': len(train_ds),
            'test_samples': len(test_ds),
            'img_size': p_p["img_size"]
        }
    }
    
    metrics_path = Path(dvc["metrics_file"])
    save_metrics(final_metrics, metrics_path)
    print(f"Model saved to {model_path}, metrics to {metrics_path}")
    print(f"Final test accuracy: {history['test_acc'][-1]:.4f}")
    print(f"Best test accuracy: {max(history['test_acc']):.4f}")

if __name__=="__main__":
    main()
