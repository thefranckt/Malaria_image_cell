# 🦠 Malaria Cell Classification

Un classificateur d'images utilisant PyTorch et ResNet18 pour détecter automatiquement la présence de parasites de malaria dans des cellules sanguines.

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-v2.0+-red.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## 🎯 Objectif

Ce projet implémente un système de classification automatique pour identifier les cellules sanguines infectées par la malaria. Il utilise des techniques de deep learning avec transfer learning sur un modèle ResNet18 pré-entraîné.

## 📊 Dataset

- **Classes** : Parasitized (infecté) / Uninfected (sain)
- **Format** : Images PNG 64x64 pixels
- **Source** : Images de cellules sanguines microscopiques
- **Split** : 80% entraînement / 20% test

## 🏗️ Architecture

```
src/
├── preprocess.py      # Préparation des données
├── train.py          # Entraînement du modèle
└── deploy.py         # Déploiement et inférence

config/
├── params.yaml       # Configuration principale
└── deploy.yaml       # Configuration déploiement

api.py                # API Flask pour déploiement web
```

## 🚀 Installation

### Prérequis
```bash
Python 3.9+
pip install -r requirements.txt
```

### Dépendances principales
```bash
torch>=2.0.1
torchvision>=0.15.2
pyyaml>=6.0.1
pillow>=10.0.0
dvc>=3.0.0
```

## 📈 Utilisation

### 1. Préparation des données
```bash
python src/preprocess.py
```

### 2. Entraînement
```bash
python src/train.py
```

### 3. Déploiement local
```bash
python api.py
```

### 4. Pipeline DVC complet
```bash
dvc repro
```

## 🎛️ Configuration

Le fichier `params.yaml` contient tous les hyperparamètres :

```yaml
train:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  
model:
  architecture: "resnet18"
  pretrained: true
  
preprocess:
  img_size: 64
  test_split: 0.2
```

## 📊 Performances

- **Architecture** : ResNet18 avec transfer learning
- **Accuracy attendue** : >95%
- **Temps d'entraînement** : ~10 minutes (10 epochs)
- **Taille du modèle** : ~45MB

## 🌐 API Web

Interface web disponible sur `http://localhost:5000` avec :

- Upload d'images via interface intuitive
- Prédictions en temps réel
- Affichage des probabilités
- API REST pour intégration

### Endpoints
```
GET  /              # Interface web
POST /predict       # Prédiction single image
POST /batch_predict # Prédiction batch
GET  /health        # Health check
```

## 🐳 Déploiement Docker

```bash
# Build
docker build -t malaria-classifier .

# Run
docker run -p 5000:5000 malaria-classifier
```

## 👤 Author
**Franckt** - *Développement initial*