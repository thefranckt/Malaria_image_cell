# Dockerfile pour déploiement
FROM python:3.9-slim

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Créer le répertoire de travail
WORKDIR /app

# Copier les requirements
COPY requirements.txt .

# Installer les dépendances Python
RUN pip install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Créer un utilisateur non-root
RUN useradd --create-home --shell /bin/bash malaria
RUN chown -R malaria:malaria /app
USER malaria

# Exposer le port
EXPOSE 5000

# Commande par défaut
CMD ["python", "api.py"]
