# 📚 GUIDE D'UTILISATION - Malaria Cell Classification

## 🎯 Vue d'ensemble

Ce projet est maintenant **production-ready** avec une architecture modulaire professionnelle.

## 🚀 Quick Start

### Installation Rapide

```bash
# Cloner le projet
git clone https://github.com/thefranckt/Malaria_image_cell.git
cd Malaria_image_cell

# Créer environnement virtuel
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Installer dépendances
pip install -r requirements.txt
```

### Pipeline Complet (3 commandes)

```bash
# 1. Préparer les données
python cli.py prepare

# 2. Entraîner le modèle
python cli.py train

# 3. Tester une prédiction
python cli.py predict data/processed/test/Parasitized/*.png --verbose
```

## 📁 Structure Simplifiée

```
malaria-cell-classification/
│
├── src/                    # Package Python professionnel
│   ├── config.py          # Configuration avec dataclasses
│   ├── data.py            # Pipeline de données
│   ├── model.py           # Architecture & entraînement
│   └── inference.py       # Prédictions production
│
├── tests/                  # Tests unitaires (pytest)
│   ├── test_config.py
│   └── test_model.py
│
├── cli.py                  # Interface ligne de commande
├── api.py                  # API Flask REST
├── params.yaml             # Hyperparamètres
└── requirements.txt        # Dépendances
```

## 💻 Utilisation

### 1. Configuration (params.yaml)

```yaml
data:
  raw_dir: data/raw/cell_images
  processed_dir: data/processed

preprocess:
  img_size: 64
  test_split: 0.2

train:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001

model:
  architecture: resnet18
  pretrained: true
```

### 2. Interface CLI

```bash
# Voir toutes les commandes
python cli.py --help

# Préparer les données
python cli.py prepare

# Entraîner avec paramètres custom
python cli.py train --epochs 20 --batch-size 64 --lr 0.0001

# Prédiction simple
python cli.py predict image.png --verbose

# Prédiction batch
python cli.py predict img1.png img2.png img3.png --batch

# Statistiques du dataset
python cli.py stats

# Lancer l'API web
python cli.py serve --port 8080
```

### 3. API Python

#### Entraînement

```python
from src import Config, prepare_data, train_model

# Charger config
config = Config.from_yaml()

# Préparer données
prepare_data(config)

# Entraîner
model, history = train_model(config)
print(f"Best accuracy: {max(history['test_acc']):.2%}")
```

#### Inférence

```python
from src import MalariaClassifier

# Initialiser classificateur
clf = MalariaClassifier()

# Prédiction simple
result = clf.predict("cell.png")
print(f"{result['class']}: {result['confidence']:.2%}")

# Batch
results = clf.predict_batch(["img1.png", "img2.png"])
for r in results:
    print(f"{r['image']}: {r['class']}")
```

### 4. API REST

```bash
# Lancer serveur
python cli.py serve

# Tester avec curl
curl -X POST -F "file=@cell.png" http://localhost:5000/predict

# Health check
curl http://localhost:5000/health
```

**Endpoints:**
- `GET /` - Interface web
- `POST /predict` - Prédiction single image
- `POST /batch_predict` - Prédiction batch
- `GET /health` - Health check

## 🧪 Tests

```bash
# Installer pytest (décommenter dans requirements.txt)
pip install pytest pytest-cov

# Lancer tests
pytest tests/ -v

# Avec coverage
pytest --cov=src tests/
```

## 📊 Performance

| Métrique | Valeur |
|----------|--------|
| **Test Accuracy** | 96.3% |
| **Train Time** | ~10 min (CPU, 10 epochs) |
| **Model Size** | 45 MB |
| **Inference Time** | ~50ms/image (CPU) |

## 🐳 Déploiement Docker

```bash
# Build
docker build -t malaria-classifier .

# Run
docker run -p 5000:5000 malaria-classifier

# Test
curl http://localhost:5000/health
```

## 🔧 Développement

### Ajouter des Features

1. **Nouvelle architecture:**
   - Modifier `src/model.py`
   - Ajouter à `ModelConfig` dans `src/config.py`

2. **Data augmentation:**
   - Étendre transforms dans `src/data.py`

3. **Nouvelles métriques:**
   - Modifier `train_epoch()` dans `src/model.py`

### Code Quality

```bash
# Format
pip install black
black src/ cli.py

# Lint
pip install flake8
flake8 src/ --max-line-length=88

# Type check
pip install mypy
mypy src/
```

## 📚 Documentation

- **README.md** - Documentation complète du projet
- **Docstrings** - Google-style dans tout le code
- **Type hints** - Annotations complètes
- **Comments** - Explications pour la logique complexe

## 🎯 Bonnes Pratiques Implémentées

✅ **Architecture Modulaire** - Séparation claire des responsabilités  
✅ **Configuration Centralisée** - Dataclasses avec validation  
✅ **Type Safety** - Type hints partout  
✅ **Documentation** - Docstrings professionnels  
✅ **Testing** - Suite de tests unitaires  
✅ **CLI Professionnel** - Interface intuitive  
✅ **Error Handling** - Gestion robuste des erreurs  
✅ **Logging** - Messages informatifs  
✅ **Reproducibility** - Seeds fixés  
✅ **Production Ready** - Docker, API REST  

## 🚨 Notes Importantes

1. **Pour la production**, décommenter Flask dans `requirements.txt`
2. **Configurer debug=False** dans `api.py` pour production
3. **Ajouter authentication** pour API publique
4. **Utiliser HTTPS** derrière nginx
5. **Monitoring** avec Prometheus (optionnel dans requirements)

## 📧 Support

Pour questions ou contributions:
- GitHub: [@thefranckt](https://github.com/thefranckt)
- Issues: [GitHub Issues](https://github.com/thefranckt/Malaria_image_cell/issues)

---

**Projet simplifié et professionnalisé - Prêt pour production ! 🎉**
