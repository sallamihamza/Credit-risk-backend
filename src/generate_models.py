"""
Script pour générer le pipeline complet de prédiction de risque de crédit
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import os

print("=== Génération du pipeline de prédiction de risque de crédit ===")

# Génération d'un dataset de démonstration
np.random.seed(42)

data = {
    'person_age': np.random.randint(18, 70, 1000),
    'person_income': np.random.randint(15000, 150000, 1000),
    'person_emp_exp': np.random.randint(0, 40, 1000),
    'loan_amnt': np.random.randint(1000, 35000, 1000),
    'loan_int_rate': np.random.uniform(5.0, 20.0, 1000),
    'loan_percent_income': np.random.uniform(0.05, 0.7, 1000),
    'cb_person_cred_hist_length': np.random.randint(0, 30, 1000),
    'credit_score': np.random.randint(300, 850, 1000),
    'person_gender': np.random.choice(['Male', 'Female'], 1000),
    'person_education': np.random.choice(['High School', 'Bachelor', 'Master', 'Doctorate'], 1000),
    'person_home_ownership': np.random.choice(['RENT', 'OWN', 'MORTGAGE', 'OTHER'], 1000),
    'loan_intent': np.random.choice(['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'], 1000),
    'previous_loan_defaults_on_file': np.random.choice(['No', 'Yes'], 1000, p=[0.8, 0.2])
}

# Génération de la variable cible
risk_probabilities = []
for i in range(1000):
    base_risk = 0.3
    if data['credit_score'][i] < 600:
        base_risk += 0.3
    elif data['credit_score'][i] < 700:
        base_risk += 0.1
    if data['loan_percent_income'][i] > 0.5:
        base_risk += 0.2
    elif data['loan_percent_income'][i] > 0.3:
        base_risk += 0.1
    if data['previous_loan_defaults_on_file'][i] == 'Yes':
        base_risk += 0.4
    if data['person_age'][i] < 25:
        base_risk += 0.1
    base_risk = min(max(base_risk, 0.05), 0.95)
    risk_probabilities.append(base_risk)

data['loan_status'] = np.random.binomial(1, risk_probabilities)
df = pd.DataFrame(data)

# Séparation des features et de la cible
X = df.drop(columns=['loan_status'])
y = df['loan_status']

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

# Prétraitement
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ]
)

# Pipeline complet
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(random_state=42))
])

# Entraînement
pipeline.fit(X, y)

# Sauvegarde du pipeline complet
os.makedirs('src/models', exist_ok=True)
joblib.dump(pipeline, 'src/models/credit_risk_pipeline.pkl')

print("Pipeline complet sauvegardé dans src/models/credit_risk_pipeline.pkl")
