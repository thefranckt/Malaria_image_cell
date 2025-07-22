# setup_github.ps1 - Script PowerShell pour configurer et pusher le projet sur GitHub

Write-Host "🚀 Configuration Git et push vers GitHub..." -ForegroundColor Green

# 1. Vérifier si Git est installé
try {
    git --version | Out-Null
    Write-Host "✅ Git est installé" -ForegroundColor Green
} catch {
    Write-Host "❌ Git n'est pas installé. Installez-le depuis https://git-scm.com/" -ForegroundColor Red
    exit 1
}

# 2. Initialiser Git (si pas déjà fait)
if (!(Test-Path ".git")) {
    Write-Host "📦 Initialisation du repository Git..." -ForegroundColor Yellow
    git init
    git branch -M main
} else {
    Write-Host "✅ Repository Git déjà initialisé" -ForegroundColor Green
}

# 3. Configuration Git utilisateur
Write-Host "👤 Configuration utilisateur Git..." -ForegroundColor Yellow
$github_name = Read-Host "Entrez votre nom GitHub"
$github_email = Read-Host "Entrez votre email GitHub"

git config user.name "$github_name"
git config user.email "$github_email"

# 4. Ajouter tous les fichiers
Write-Host "📁 Ajout des fichiers..." -ForegroundColor Yellow
git add .

# 5. Commit initial
Write-Host "💾 Commit initial..." -ForegroundColor Yellow
$commit_message = @"
🎉 Initial commit: Malaria Cell Classification with PyTorch

✨ Features:
- ResNet18 transfer learning for malaria detection
- DVC pipeline for data versioning
- Flask API for deployment
- Docker support
- Comprehensive preprocessing and training scripts

🚀 Ready for deployment and further development!
"@

git commit -m $commit_message

# 6. Configuration du remote GitHub
Write-Host ""
Write-Host "📡 Configuration du remote GitHub..." -ForegroundColor Yellow
Write-Host "Créez d'abord un repository sur GitHub: https://github.com/new" -ForegroundColor Cyan
Write-Host "Nom suggéré: malaria-cell-classification" -ForegroundColor Cyan
Write-Host ""
$github_url = Read-Host "Entrez l'URL de votre repository GitHub (https://github.com/username/repo.git)"

git remote add origin $github_url

# 7. Push vers GitHub
Write-Host "🚀 Push vers GitHub..." -ForegroundColor Yellow
try {
    git push -u origin main
    Write-Host ""
    Write-Host "🎉 Projet pushé avec succès sur GitHub!" -ForegroundColor Green
    Write-Host "🌐 Votre repository: $github_url" -ForegroundColor Cyan
} catch {
    Write-Host "❌ Erreur lors du push. Vérifiez vos credentials GitHub." -ForegroundColor Red
    Write-Host "💡 Vous pouvez configurer un token: https://github.com/settings/tokens" -ForegroundColor Yellow
}

Write-Host ""
Write-Host "📋 Prochaines étapes:" -ForegroundColor Yellow
Write-Host "1. Ajoutez une description sur GitHub" -ForegroundColor White
Write-Host "2. Configurez les topics: machine-learning, pytorch, malaria, deep-learning" -ForegroundColor White
Write-Host "3. Activez GitHub Pages si vous voulez une demo" -ForegroundColor White
Write-Host "4. Configurez GitHub Actions pour CI/CD (optionnel)" -ForegroundColor White

# Pause pour lire les instructions
Read-Host "Appuyez sur Entrée pour continuer..."
