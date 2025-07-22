#!/bin/bash
# deploy.sh - Script de déploiement automatisé

echo "🚀 Déploiement du classificateur de malaria..."

# 1. Vérifier que le modèle existe
if [ ! -f "models/model.pth" ]; then
    echo "❌ Modèle non trouvé. Lancez d'abord l'entraînement:"
    echo "   python src/train.py"
    exit 1
fi

# 2. Installer les dépendances de déploiement
echo "📦 Installation des dépendances..."
pip install -r requirements_deploy.txt

# 3. Tester le classificateur
echo "🧪 Test du classificateur..."
python src/deploy.py

# 4. Option: Déploiement local
echo "🌐 Démarrage du serveur local..."
echo "   API disponible sur: http://localhost:5000"
echo "   Health check: http://localhost:5000/health"
echo ""
echo "Appuyez sur Ctrl+C pour arrêter"

# Démarrer l'API Flask
python api.py
