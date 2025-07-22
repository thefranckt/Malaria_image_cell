#!/bin/bash
# setup_github.sh - Script pour configurer et pusher le projet sur GitHub

echo "🚀 Configuration Git et push vers GitHub..."

# 1. Initialiser Git (si pas déjà fait)
if [ ! -d ".git" ]; then
    echo "📦 Initialisation du repository Git..."
    git init
    git branch -M main
else
    echo "✅ Repository Git déjà initialisé"
fi

# 2. Configuration Git utilisateur (à personnaliser)
echo "👤 Configuration utilisateur Git..."
read -p "Entrez votre nom GitHub: " github_name
read -p "Entrez votre email GitHub: " github_email

git config user.name "$github_name"
git config user.email "$github_email"

# 3. Ajouter tous les fichiers
echo "📁 Ajout des fichiers..."
git add .

# 4. Commit initial
echo "💾 Commit initial..."
git commit -m "🎉 Initial commit: Malaria Cell Classification with PyTorch

✨ Features:
- ResNet18 transfer learning for malaria detection
- DVC pipeline for data versioning
- Flask API for deployment
- Docker support
- Comprehensive preprocessing and training scripts

🚀 Ready for deployment and further development!"

# 5. Demander l'URL du repository GitHub
echo ""
echo "📡 Configuration du remote GitHub..."
echo "Créez d'abord un repository sur GitHub: https://github.com/new"
echo "Nom suggéré: malaria-cell-classification"
echo ""
read -p "Entrez l'URL de votre repository GitHub (https://github.com/username/repo.git): " github_url

git remote add origin "$github_url"

# 6. Push vers GitHub
echo "🚀 Push vers GitHub..."
git push -u origin main

echo ""
echo "🎉 Projet pushé avec succès sur GitHub!"
echo "🌐 Votre repository: $github_url"
echo ""
echo "📋 Prochaines étapes:"
echo "1. Ajoutez une description sur GitHub"
echo "2. Configurez les topics: machine-learning, pytorch, malaria, deep-learning"
echo "3. Activez GitHub Pages si vous voulez une demo"
echo "4. Configurez GitHub Actions pour CI/CD (optionnel)"
