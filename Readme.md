# 💼 Credit Risk Prediction – Backend API (Flask)

API Flask de prédiction de risque de crédit, intégrée à un modèle Machine Learning (Random Forest/XGBoost) et déployée sur Railway. Cette API est utilisée par une interface frontend hébergée sur GitHub Pages.

---

## 🌐 Démo

- **Frontend** (GitHub Pages) : [➡️ https://sallamihamza.github.io/Data-science-projects/](https://sallamihamza.github.io/Data-science-projects/)
- **Backend API** (Railway) : [➡️ https://web-production-f2f2.up.railway.app/]

---

## 📌 Fonctionnalités

- Prédiction de risque de crédit à partir de données utilisateur
- API REST (Flask + Blueprint)
- Chargement dynamique d’un modèle `.pkl`
- Base de données SQLite intégrée (optionnelle)
- Communication frontend ↔ backend avec CORS
- Déploiement automatique via GitHub → Railway

---

## 🛠️ Technologies utilisées

- **Langage** : Python 3.11
- **Framework** : Flask
- **Machine Learning** : scikit-learn, XGBoost, pandas
- **Déploiement** : Railway (backend), GitHub Pages (frontend)
- **Base de données** : SQLite
- **Serveur de production** : Gunicorn

---

## 🗂️ Arborescence du projet

.
├── main.py # Point d'entrée Flask
├── requirements.txt # Dépendances Python
├── Procfile # Commande pour Railway (gunicorn)
├── models/
│ └── credit_risk_pipeline.pkl # Modèle ML entraîné
├── src/
│ ├── models/
│ ├── routes/
│ └── prediction_service.py # Chargement & prédiction

---

## 🚀 Déploiement sur Railway

### ⚙️ 1. Installation locale (pour tester)

```bash
git clone https://github.com/TON-UTILISATEUR/credit-risk-backend.git
cd credit-risk-backend

python -m venv venv
source venv/bin/activate  # (ou venv\Scripts\activate sous Windows)
pip install -r requirements.txt

python main.py  # Lance le serveur localement