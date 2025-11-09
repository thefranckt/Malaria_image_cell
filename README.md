# Malaria Cell Image Classification

> **Production-ready deep learning system for automated malaria parasite detection in microscopic blood cell images**

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: professional](https://img.shields.io/badge/code%20style-professional-brightgreen.svg)]()

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Model Performance](#model-performance)
- [Usage](#usage)
  - [CLI Interface](#cli-interface)
  - [Python API](#python-api)
  - [REST API](#rest-api)
- [Configuration](#configuration)
- [Docker Deployment](#docker-deployment)
- [Development](#development)
- [Dataset](#dataset)
- [License](#license)
- [Author](#author)

---

## 🎯 Overview

This project implements a **transfer learning-based computer vision system** for binary classification of malaria-infected blood cells. Using ResNet18 pre-trained on ImageNet, the model achieves **96.3% accuracy** on microscopic cell images.

### Clinical Context

Malaria diagnosis via microscopy is labor-intensive and requires trained personnel. This automated system provides:

- **Fast, scalable screening** for large sample volumes
- **Consistent, reproducible results** across different operators
- **Decision support tool** for medical professionals
- **Deployment potential** in resource-limited settings

### Technical Stack

- **Framework**: PyTorch 2.0+ with torchvision
- **Architecture**: ResNet18 (transfer learning from ImageNet)
- **Input**: 64×64 RGB microscopic cell images
- **Output**: Binary classification (Parasitized / Uninfected)
- **Deployment**: Flask REST API with Docker containerization

---

## ✨ Key Features

### Engineering Best Practices

✅ **Modular Architecture**: Clean separation of concerns (config / data / model / inference)  
✅ **Type-Safe**: Full type hints and dataclass-based configuration  
✅ **Well-Documented**: Google-style docstrings throughout  
✅ **Production-Ready**: CLI tool, REST API, Docker deployment  
✅ **Tested**: Unit tests with pytest framework  
✅ **Version Controlled**: Git + DVC for code and data versioning

### ML Pipeline

- **Data augmentation ready** (extensible transforms)
- **Reproducible experiments** (fixed random seeds)
- **Metric tracking** (train/test loss & accuracy history)
- **Model checkpointing** (automatic best model saving)
- **Efficient data loading** (PyTorch DataLoader optimizations)

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/thefranckt/Malaria_image_cell.git
cd Malaria_image_cell

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Basic Workflow

```bash
# 1. Prepare data (resize, split train/test)
python cli.py prepare

# 2. Train model
python cli.py train

# 3. Make predictions
python cli.py predict path/to/cell_image.png --verbose

# 4. View dataset statistics
python cli.py stats

# 5. Launch web API
python cli.py serve --port 5000
```

---

## 📁 Project Structure

```
malaria-cell-classification/
│
├── src/                          # Core Python package
│   ├── __init__.py              # Package exports
│   ├── config.py                # Configuration management (dataclasses)
│   ├── data.py                  # Data preprocessing pipeline
│   ├── model.py                 # Model architecture & training
│   └── inference.py             # Inference engine
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_config.py           # Configuration tests
│   └── test_model.py            # Model architecture tests
│
├── data/                         # Data directory
│   ├── raw/                     # Original images (DVC tracked)
│   └── processed/               # Preprocessed images
│       ├── train/              # 80% training set
│       └── test/               # 20% test set
│
├── models/                       # Trained models
│   └── model.pth                # Latest checkpoint
│
├── config/                       # Configuration files
│   └── deploy.yaml              # Deployment configuration
│
├── cli.py                        # Command-line interface
├── api.py                        # Flask REST API
├── params.yaml                   # Hyperparameters
├── requirements.txt              # Python dependencies
├── Dockerfile                    # Container configuration
├── dvc.yaml                      # DVC pipeline definition
└── README.md                     # This file
```

---

## 📊 Model Performance

### Metrics

| Metric | Value |
|--------|-------|
| **Test Accuracy** | **96.3%** |
| Train Accuracy | 99.8% |
| Test Loss | 0.196 |
| Training Time | ~10 min (10 epochs, CPU) |
| Model Size | ~45 MB |
| Inference Speed | ~50ms/image (CPU) |

### Architecture

**Model:** ResNet18 (ImageNet pretrained) with modified final layer

```
Input (64×64 RGB)
      ↓
ResNet18 (ImageNet pretrained)
      ↓
Fully Connected (512 → 2)
      ↓
Softmax
      ↓
Output: [P(Parasitized), P(Uninfected)]
```

**Key Design Decisions:**
- **Transfer Learning**: ResNet18 pretrained on ImageNet provides strong feature extraction
- **Fine-tuning**: Only final FC layer modified (faster training, reduced overfitting)
- **Image Size**: 64×64 balances detail preservation with computational efficiency
- **Normalization**: ImageNet statistics (standard for transfer learning)

---

## 💻 Usage

### CLI Interface

```bash
# Prepare data (resize images, create train/test split)
python cli.py prepare [--quiet]

# Train model with custom parameters
python cli.py train \
    [--epochs EPOCHS] \
    [--batch-size SIZE] \
    [--lr LEARNING_RATE] \
    [--quiet]

# Single image prediction
python cli.py predict cell_image.png --verbose

# Batch prediction
python cli.py predict img1.png img2.png img3.png --batch

# View dataset statistics
python cli.py stats

# Launch REST API server
python cli.py serve [--host HOST] [--port PORT] [--debug]
```

### Python API

#### Training

```python
from src import Config, prepare_data, train_model

# Load configuration
config = Config.from_yaml("params.yaml")

# Prepare data
prepare_data(config, verbose=True)

# Train model
model, history = train_model(config, verbose=True)

# Check best accuracy
print(f"Best test accuracy: {max(history['test_acc']):.4f}")
```

#### Inference

```python
from src import MalariaClassifier

# Initialize classifier
classifier = MalariaClassifier(model_path="models/model.pth")

# Single prediction
result = classifier.predict("cell_image.png")
print(f"Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
results = classifier.predict_batch([
    "image1.png",
    "image2.png",
    "image3.png"
])

for r in results:
    print(f"{r['image']}: {r['class']} ({r['confidence']:.1%})")
```

### REST API

#### Start Server

```bash
# Using CLI
python cli.py serve --port 5000

# Or directly
python api.py
```

#### Endpoints

**`GET /`** - Web interface for image upload

**`POST /predict`** - Single image classification

```bash
curl -X POST -F "file=@cell_image.png" http://localhost:5000/predict
```

Response:
```json
{
  "class": "Parasitized",
  "confidence": 0.982,
  "probabilities": {
    "Parasitized": 0.982,
    "Uninfected": 0.018
  }
}
```

**`POST /batch_predict`** - Batch image classification

```bash
curl -X POST \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  http://localhost:5000/batch_predict
```

**`GET /health`** - Health check

```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

---

## ⚙️ Configuration

Edit `params.yaml` to customize hyperparameters:

```yaml
data:
  raw_dir: data/raw/cell_images
  processed_dir: data/processed

preprocess:
  img_size: 64
  test_split: 0.2
  random_seed: 42

train:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0001

model:
  architecture: resnet18
  pretrained: true
```

---

## 🐳 Docker Deployment

### Build and Run

```bash
# Build image
docker build -t malaria-classifier .

# Run container
docker run -p 5000:5000 malaria-classifier

# Test health endpoint
curl http://localhost:5000/health
```

### Production Considerations

- **Monitoring**: Add Prometheus metrics for production tracking
- **Logging**: Configure structured logging (e.g., JSON logs)
- **Authentication**: Implement API key validation
- **Rate Limiting**: Use Flask-Limiter to prevent abuse
- **HTTPS**: Deploy behind nginx with SSL certificates
- **Scaling**: Use gunicorn with multiple workers

### Cloud Deployment

The Docker image can be deployed to:
- **AWS**: ECS, EC2, SageMaker
- **GCP**: Cloud Run, GKE, AI Platform
- **Azure**: Container Instances, AKS, ML Service
- **Heroku**: Container registry

---

## 🛠️ Development

### Code Quality

```bash
# Format code (uncomment black in requirements.txt first)
black src/ cli.py api.py

# Lint code (uncomment flake8)
flake8 src/ --max-line-length=88

# Type checking (uncomment mypy)
mypy src/
```

### Testing

```bash
# Install test dependencies (uncomment pytest in requirements.txt)
pip install pytest pytest-cov

# Run tests
pytest tests/ -v

# With coverage report
pytest --cov=src tests/
```

### Adding New Features

The codebase is designed for extensibility:

1. **New architectures**: Add to `src/model.py`, update `config.py`
2. **Data augmentation**: Extend transforms in `src/data.py`
3. **Metrics**: Modify `train_epoch()` and `evaluate()` functions
4. **API endpoints**: Add routes to `api.py`

---

## 📚 Dataset

- **Source**: [Malaria Cell Images Dataset (Kaggle)](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
- **Classes**: 
  - `Parasitized` - Cells infected with malaria parasites
  - `Uninfected` - Healthy blood cells
- **Total Images**: ~27,500 microscopic images
- **Format**: PNG, thin blood smear images
- **Split**: 80% training / 20% testing (stratified)

---

## 📄 License

This project is licensed under the **MIT License** - see [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Franckt**  
- GitHub: [@thefranckt](https://github.com/thefranckt)
- Project: [Malaria_image_cell](https://github.com/thefranckt/Malaria_image_cell)

---

## 🙏 Acknowledgments

- Dataset provided by Kaggle community
- PyTorch team for excellent deep learning framework
- ResNet authors: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

---

## ⚠️ Disclaimer

**This is a research/educational project.** For clinical deployment, rigorous validation and regulatory compliance (FDA, CE marking, etc.) are required. This tool should be used as a **decision support system** under supervision of qualified medical professionals, not as a standalone diagnostic tool.

---

## 📖 Citation

If you use this project in your research, please cite:

```bibtex
@software{malaria_cell_classifier_2025,
  author = {Franckt},
  title = {Malaria Cell Image Classification},
  year = {2025},
  url = {https://github.com/thefranckt/Malaria_image_cell}
}
```

## 📊 Performance

### **Model Metrics**

| Metric | Value |
|--------|-------|
| **Test Accuracy** | 96.3% |
| **Train Accuracy** | 99.8% |
| **Test Loss** | 0.196 |
| **Training Time** | ~10 min (10 epochs, CPU) |
| **Model Size** | ~45 MB |

### **Training Curves**

The model shows **excellent convergence** with minimal overfitting:
- Accuracy plateaus around epoch 5-6
- Validation loss remains stable
- Final test accuracy: **96.30%**

### **Confusion Matrix** (Expected Performance)

|                | Predicted: Parasitized | Predicted: Uninfected |
|----------------|:----------------------:|:---------------------:|
| **Actual: Parasitized** | ~2650 (96%) | ~105 (4%) |
| **Actual: Uninfected**  | ~100 (4%) | ~2655 (96%) |

## 💻 Usage

### **Configuration**

Edit `params.yaml` to customize hyperparameters:

```yaml
data:
  raw_dir: data/raw/cell_images
  processed_dir: data/processed

preprocess:
  img_size: 64
  test_split: 0.2
  random_seed: 42

train:
  batch_size: 32
  epochs: 10
  learning_rate: 0.001
  momentum: 0.9
  weight_decay: 0.0001

model:
  architecture: resnet18
  pretrained: true
```

### **Python API Examples**

#### **Training**

```python
from src import Config, prepare_data, train_model

# Load configuration
config = Config.from_yaml("params.yaml")

# Prepare data
prepare_data(config, verbose=True)

# Train model
model, history = train_model(config, verbose=True)

# Check best accuracy
print(f"Best test accuracy: {max(history['test_acc']):.4f}")
```

#### **Inference**

```python
from src import MalariaClassifier

# Initialize classifier
classifier = MalariaClassifier(model_path="models/model.pth")

# Single prediction
result = classifier.predict("cell_image.png")
print(f"Class: {result['class']}")
print(f"Confidence: {result['confidence']:.2%}")

# Batch prediction
results = classifier.predict_batch([
    "image1.png",
    "image2.png",
    "image3.png"
])

for r in results:
    print(f"{r['image']}: {r['class']} ({r['confidence']:.1%})")
```

### **CLI Reference**

```bash
# Prepare data
python cli.py prepare [--quiet]

# Train model
python cli.py train \\
    [--epochs EPOCHS] \\
    [--batch-size SIZE] \\
    [--lr LEARNING_RATE] \\
    [--quiet]

# Make predictions
python cli.py predict IMAGE [IMAGE ...] \\
    [--model MODEL_PATH] \\
    [--batch] \\
    [--verbose]

# View statistics
python cli.py stats

# Launch API server
python cli.py serve \\
    [--host HOST] \\
    [--port PORT] \\
    [--debug]
```

## 🌐 API Documentation

### **REST Endpoints**

#### `GET /`
Web interface for image upload and classification

#### `POST /predict`
Single image classification

**Request**:
```bash
curl -X POST -F "file=@cell_image.png" http://localhost:5000/predict
```

**Response**:
```json
{
  "class": "Parasitized",
  "confidence": 0.982,
  "probabilities": {
    "Parasitized": 0.982,
    "Uninfected": 0.018
  }
}
```

#### `POST /batch_predict`
Batch image classification

**Request**:
```bash
curl -X POST \\
  -F "files=@image1.png" \\
  -F "files=@image2.png" \\
  http://localhost:5000/batch_predict
```

#### `GET /health`
Health check endpoint

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true
}
```

## 🛠️ Development

### **Code Quality**

```bash
# Format code (uncomment black in requirements.txt)
black src/ cli.py api.py

# Lint code (uncomment flake8)
flake8 src/ --max-line-length=88

# Type checking (uncomment mypy)
mypy src/
```

### **Testing**

```bash
# Run tests (uncomment pytest)
pytest tests/ -v

# With coverage
pytest --cov=src tests/
```

### **Adding New Features**

The codebase is designed for extensibility:

1. **New architectures**: Add to `src/model.py`, update `config.py`
2. **Data augmentation**: Extend transforms in `src/data.py`
3. **Metrics**: Add to `train_epoch()` and `evaluate()` functions
4. **API endpoints**: Add routes to `api.py`

## 🐳 Deployment

### **Docker**

```bash
# Build image
docker build -t malaria-classifier .

# Run container
docker run -p 5000:5000 malaria-classifier

# Access API
curl http://localhost:5000/health
```

### **Production Considerations**

- **Monitoring**: Add Prometheus metrics (commented in requirements.txt)
- **Logging**: Configure structured logging with `structlog`
- **Authentication**: Implement API key validation
- **Rate Limiting**: Use Flask-Limiter
- **HTTPS**: Deploy behind nginx with SSL
- **Scaling**: Use gunicorn with multiple workers

### **Cloud Deployment**

The Docker image can be deployed to:
- **AWS**: ECS, EC2, SageMaker
- **GCP**: Cloud Run, GKE
- **Azure**: Container Instances, AKS
- **Heroku**: Container registry

## 📚 References

### **Dataset**
- Source: [Malaria Cell Images Dataset](https://www.kaggle.com/iarunava/cell-images-for-detecting-malaria)
- Classes: Parasitized (infected) / Uninfected (healthy)
- Total images: ~27,000 cells
- Format: PNG, microscopic thin blood smears

### **Model**
- ResNet: [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)
- Transfer Learning: [CS231n Guide](https://cs231n.github.io/transfer-learning/)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

## 👤 Author

**Franckt**
- GitHub: [@thefranckt](https://github.com/thefranckt)
- Project Link: [https://github.com/thefranckt/Malaria_image_cell](https://github.com/thefranckt/Malaria_image_cell)

## 🙏 Acknowledgments

- Dataset provided by Kaggle community
- PyTorch team for excellent framework
- ResNet authors for foundational architecture

---

**Note**: This is a research/educational project. For clinical deployment, rigorous validation and regulatory compliance (FDA, CE marking, etc.) are required.
