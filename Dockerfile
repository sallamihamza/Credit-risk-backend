# Utilise une image Python officielle légère
FROM python:3.11-slim

# Définit le répertoire de travail dans le container
WORKDIR /app

# Copie seulement requirements.txt d'abord (cache docker)
COPY requirements.txt .

# Installe les dépendances
RUN pip install --no-cache-dir -r requirements.txt

# Copie tout le code source
COPY . .

# Expose le port sur lequel Flask écoute (par défaut 5000)
EXPOSE 5000

# Définit la variable d'environnement pour Flask (si besoin)
ENV FLASK_APP=src/main.py
ENV FLASK_RUN_HOST=0.0.0.0

# Commande pour lancer l'application Flask avec Gunicorn (production)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "src.main:app"]
