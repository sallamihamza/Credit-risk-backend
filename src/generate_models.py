"""
Script pour générer le pipeline complet de prédiction de risque de crédit
"""

import os
import joblib
import pandas as pd
import numpy as np  # Import manquant
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score

print("=== Génération du pipeline de prédiction de risque de crédit ===")
print(f"Version de scikit-learn: {__import__('sklearn').__version__}")

# Génération d'un dataset de démonstration
np.random.seed(42)

print("Génération des données de démonstration...")

data = {
    'person_age': np.random.randint(18, 70, 1000),
    'person_income': np.random.randint(15000, 150000, 1000),
    'person_emp_exp': np.random.randint(0, 40, 1000),
    'loan_amnt': np.random.randint(1000, 35000, 1000),
    'loan_int_rate': np.round(np.random.uniform(5.0, 20.0, 1000), 2),
    'loan_percent_income': np.round(np.random.uniform(0.05, 0.7, 1000), 3),
    'cb_person_cred_hist_length': np.random.randint(0, 30, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'person_gender': np.random.choice(['Male', 'Female'], 1000),
    'person_education': np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], 1000),
    'person_home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], 1000),
    'loan_intent': np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], 1000),
    'previous_loan_defaults_on_file': np.random.choice(['No', 'Yes'], 1000, p=[0.8, 0.2])
}

print("Génération des probabilités de risque...")

# Génération de la variable cible avec logique métier
risk_probabilities = []
for i in range(1000):
    base_risk = 0.3
    
    # Impact du score de crédit
    if data['credit_score'][i] < 600:
        base_risk += 0.3
    elif data['credit_score'][i] < 700:
        base_risk += 0.1
    
    # Impact du ratio revenu/prêt
    if data['loan_percent_income'][i] > 0.5:
        base_risk += 0.2
    elif data['loan_percent_income'][i] > 0.3:
        base_risk += 0.1
    
    # Impact des défauts précédents
    if data['previous_loan_defaults_on_file'][i] == 'Yes':
        base_risk += 0.4
    
    # Impact de l'âge
    if data['person_age'][i] < 25:
        base_risk += 0.1
    
    # Impact de l'expérience professionnelle
    if data['person_emp_exp'][i] < 2:
        base_risk += 0.15
    
    # Impact du taux d'intérêt (indicateur de risque)
    if data['loan_int_rate'][i] > 15:
        base_risk += 0.1
    
    # Limiter entre 5% et 95%
    base_risk = min(max(base_risk, 0.05), 0.95)
    risk_probabilities.append(base_risk)

# Génération de la variable cible binaire
data['loan_status'] = np.random.binomial(1, risk_probabilities)

# Création du DataFrame
df = pd.DataFrame(data)

print(f"Dataset généré: {df.shape}")
print(f"Distribution de la cible: {df['loan_status'].value_counts().to_dict()}")

# Séparation des features et de la cible
X = df.drop(columns=['loan_status'])
y = df['loan_status']

# Division train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Taille d'entraînement: {X_train.shape}")
print(f"Taille de test: {X_test.shape}")

# Définition des colonnes
categorical_features = [
    'person_gender',
    'person_education',
    'person_home_ownership',
    'loan_intent',
    'previous_loan_defaults_on_file'
]

numeric_features = [
    'person_age',
    'person_income',
    'person_emp_exp',
    'loan_amnt',
    'loan_int_rate',
    'loan_percent_income',
    'cb_person_cred_hist_length',
    'credit_score'
]

print("Création du préprocesseur...")

# Prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', drop='first'), categorical_features)
    ],
    remainder='drop'
)

print("Création du pipeline...")

# Pipeline complet avec hyperparamètres optimisés
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        class_weight='balanced'
    ))
])

print("Entraînement du modèle...")

# Entraînement
pipeline.fit(X_train, y_train)

print("Évaluation du modèle...")

# Évaluation
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Précision sur le test: {accuracy:.3f}")
print("\nRapport de classification:")
print(classification_report(y_test, y_pred))

# Création du dossier de destination
os.makedirs('src/models', exist_ok=True)

print("Sauvegarde du pipeline...")

# Sauvegarde du pipeline complet
pipeline_path = 'src/models/credit_risk_pipeline.pkl'
joblib.dump(pipeline, pipeline_path)

print(f"✅ Pipeline complet sauvegardé dans {pipeline_path}")

# Vérification de la sauvegarde
try:
    loaded_pipeline = joblib.load(pipeline_path)
    test_prediction = loaded_pipeline.predict(X_test[:1])
    print(f"✅ Vérification: Pipeline rechargé avec succès, prédiction test: {test_prediction[0]}")
except Exception as e:
    print(f"❌ Erreur lors du rechargement: {e}")

print("=== Génération terminée ===")

# Affichage des informations sur les features
print(f"\nFeatures numériques ({len(numeric_features)}): {numeric_features}")
print(f"Features catégorielles ({len(categorical_features)}): {categorical_features}")

# Sauvegarde des métadonnées
metadata = {
    'numeric_features': numeric_features,
    'categorical_features': categorical_features,
    'model_type': 'RandomForestClassifier',
    'sklearn_version': __import__('sklearn').__version__,
    'accuracy': accuracy,
    'n_samples': len(df)
}

metadata_path = 'src/models/model_metadata.json'
import json
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)

print(f"✅ Métadonnées sauvegardées dans {metadata_path}")