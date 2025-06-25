import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
import numpy as np

# --- Fonctions de l'Agent IA (adaptées de ai_agent.py et du notebook) ---
# Charger le modèle entraîné et le scaler
@st.cache_resource # Mettre en cache le chargement pour éviter de recharger à chaque exécution
def load_resources():
    try:
        model = joblib.load("best_credit_risk_model.pkl")
        scaler = joblib.load("scaler.pkl")
        feature_names_df = pd.read_csv("feature_names.csv")
        feature_names = feature_names_df["feature_name"].tolist()
        # Charger un échantillon de X_train pour obtenir toutes les colonnes après prétraitement
        X_train_sample = pd.read_csv("X_train.csv", nrows=1)
        all_model_columns = X_train_sample.columns.tolist()
        return model, scaler, feature_names, all_model_columns
    except FileNotFoundError:
        st.error("Fichiers de modèle/scaler/features non trouvés. Assurez-vous d'avoir exécuté les étapes précédentes du notebook.")
        st.stop()

model, scaler, feature_names, all_model_columns = load_resources()

def predict_risk_streamlit(client_data_dict):
    # Convertir les données d'entrée en DataFrame
    client_df = pd.DataFrame([client_data_dict])

    # Définir les colonnes numériques et catégorielles (doit correspondre au prétraitement)
    numerical_cols = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
    categorical_cols = ["person_gender", "person_education", "person_home_ownership", "loan_intent", "previous_loan_defaults_on_file"]

    # Appliquer One-Hot Encoding
    client_df_processed = pd.get_dummies(client_df, columns=categorical_cols, drop_first=True)

    # Assurer que toutes les colonnes du modèle sont présentes (remplir avec 0 si manquant)
    for col in all_model_columns:
        if col not in client_df_processed.columns:
            client_df_processed[col] = 0
    
    # Réordonner les colonnes pour correspondre à l'ordre d'entraînement du modèle
    client_df_processed = client_df_processed[all_model_columns]

    # Appliquer le scaler aux colonnes numériques
    client_df_processed[numerical_cols] = scaler.transform(client_df_processed[numerical_cols])

    # Prédiction de la probabilité de risque
    risk_proba = model.predict_proba(client_df_processed)[:, 1][0]
    return risk_proba

def explain_risk_streamlit(client_data, risk_proba):
    explanation = []
    actions = []

    # Règles basées sur la probabilité de risque
    if risk_proba >= 0.7:
        explanation.append("Le risque est **très élevé**.")
        actions.append("Lancer immédiatement une procédure de recouvrement accélérée.")
        actions.append("Contacter le client pour comprendre la situation et négocier un plan de paiement.")
    elif risk_proba >= 0.5:
        explanation.append("Le risque est **élevé**.")
        actions.append("Mettre le client sous surveillance étroite.")
        actions.append("Envoyer un rappel de paiement préventif.")
    elif risk_proba >= 0.3:
        explanation.append("Le risque est **modéré**.")
        actions.append("Surveiller le comportement de paiement du client.")
        actions.append("Proposer des options de paiement flexibles si nécessaire.")
    else:
        explanation.append("Le risque est **faible**.")
        actions.append("Maintenir une relation client normale.")

    # Règles basées sur les caractéristiques du client (adaptées du notebook)
    if client_data["person_income"] < 30000:
        explanation.append(f'Un revenu annuel de {client_data["person_income"]} € est relativement faible, ce qui peut augmenter le risque.')
    if client_data["loan_percent_income"] > 0.4:
        explanation.append(f"Le pourcentage du prêt par rapport au revenu ({client_data["loan_percent_income"]:.1%}) est élevé, indiquant une charge financière importante.")
    if client_data["cb_person_cred_hist_length"] < 3:
        explanation.append(f"L'historique de crédit court ({client_data["cb_person_cred_hist_length"]} ans) peut rendre l'évaluation plus incertaine.")
    if client_data["credit_score"] < 600:
        explanation.append(f"Un score de crédit de {client_data["credit_score"]} est faible, ce qui est un indicateur clé de risque.")
    if client_data["previous_loan_defaults_on_file"] == "Yes":
        explanation.append("Des défauts de paiement précédents ont été enregistrés, ce qui est un facteur de risque majeur.")
    if client_data["person_emp_exp"] < 1:
        explanation.append(f"L'expérience professionnelle ({client_data["person_emp_exp"]} ans) est très faible, ce qui peut impacter la stabilité financière.")

    return " ".join(explanation), " ".join(actions)

# --- Interface Streamlit ---
st.set_page_config(layout="wide")
st.title("Prédiction du Risque Client et Agent IA pour iMX")

st.sidebar.header("Informations Client")

with st.sidebar.form("client_form"):
    person_age = st.number_input("Âge", min_value=18, max_value=100, value=30)
    person_gender = st.selectbox("Genre", ["Male", "Female"])
    person_education = st.selectbox("Niveau d'éducation", ["High School", "Bachelor", "Master", "Doctorate", "Associate"])
    person_income = st.number_input("Revenu Annuel (€)", min_value=0, value=50000)
    person_emp_exp = st.number_input("Années d'expérience professionnelle", min_value=0, max_value=50, value=5)
    person_home_ownership = st.selectbox("Propriété du logement", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    loan_amnt = st.number_input("Montant du prêt (€)", min_value=0, value=15000)
    loan_intent = st.selectbox("Intention du prêt", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])
    loan_int_rate = st.number_input("Taux d'intérêt du prêt (%)", min_value=0.0, max_value=30.0, value=10.5)
    loan_percent_income = st.slider("Pourcentage du prêt par rapport au revenu", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
    cb_person_cred_hist_length = st.number_input("Ancienneté de l'historique de crédit (années)", min_value=0, max_value=50, value=7)
    credit_score = st.number_input("Score de crédit", min_value=300, max_value=850, value=680)
    previous_loan_defaults_on_file = st.selectbox("Défauts de paiement précédents ?", ["No", "Yes"])

    submitted = st.form_submit_button("Prédire le Risque")

    if submitted:
        client_data = {
            "person_age": person_age,
            "person_gender": person_gender,
            "person_education": person_education,
            "person_income": person_income,
            "person_emp_exp": person_emp_exp,
            "person_home_ownership": person_home_ownership,
            "loan_amnt": loan_amnt,
            "loan_intent": loan_intent,
            "loan_int_rate": loan_int_rate,
            "loan_percent_income": loan_percent_income,
            "cb_person_cred_hist_length": cb_person_cred_hist_length,
            "credit_score": credit_score,
            "previous_loan_defaults_on_file": previous_loan_defaults_on_file
        }

        risk_proba = predict_risk_streamlit(client_data)
        explanation, actions = explain_risk_streamlit(client_data, risk_proba)

        st.subheader("Résultats de la Prédiction")
        st.write(f"Probabilité de risque d'impayé : **{risk_proba:.2f}**")

        # Jauge de risque
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = risk_proba * 100,
            title = {"text": "Niveau de Risque"},
            domain = {"x": [0, 1], "y": [0, 1]},
            gauge = {
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 30], "color": "lightgreen"},
                    {"range": [30, 50], "color": "yellow"},
                    {"range": [50, 70], "color": "orange"},
                    {"range": [70, 100], "color": "red"}
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": 50
                }
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Explication de l'Agent IA")
        st.info(explanation)

        st.subheader("Actions Recommandées par l'Agent IA")
        st.success(actions)

# --- Visualisations Générales des Données (dans le corps principal) ---
st.header("Analyse Générale du Dataset")

# Charger le dataset complet pour les visualisations exploratoires
try:
    df_full = pd.read_csv("loan_data.csv") # Charger le dataset réel original
    df_full = df_full.rename(columns={"loan_status": "target"})

    # Distribution de la variable cible
    st.subheader("Distribution du Risque Client")
    fig_target = px.histogram(df_full, x="target", title="Distribution de la Variable Cible (0=Payeur, 1=Impayé)")
    st.plotly_chart(fig_target, use_container_width=True)

    # Taux de risque global
    risk_clients = len(df_full[df_full["target"] == 1])
    total_clients = len(df_full)
    risk_rate = (risk_clients / total_clients * 100)
    st.metric("Taux de risque global dans le dataset", f"{risk_rate:.1f}%")

    # Matrice de corrélation des variables numériques
    st.subheader("Matrice de Corrélation des Variables Numériques")
    numerical_cols_full = ["person_age", "person_income", "person_emp_exp", "loan_amnt", "loan_int_rate", "loan_percent_income", "cb_person_cred_hist_length", "credit_score"]
    corr_matrix = df_full[numerical_cols_full].corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, aspect="auto", color_continuous_scale="RdBu", title="Matrice de Corrélation")
    st.plotly_chart(fig_corr, use_container_width=True)

    # Distributions de quelques variables clés
    st.subheader("Distributions des Variables Clés")
    fig_dist = px.histogram(df_full, x="person_income", title="Distribution du Revenu Annuel")
    st.plotly_chart(fig_dist, use_container_width=True)

    fig_dist_age = px.histogram(df_full, x="person_age", title="Distribution de l'Âge")
    st.plotly_chart(fig_dist_age, use_container_width=True)

    fig_dist_loan = px.histogram(df_full, x="loan_amnt", title="Distribution du Montant du Prêt")
    st.plotly_chart(fig_dist_loan, use_container_width=True)

except FileNotFoundError:
    st.warning("Le fichier 'loan_data.csv' n'a pas été trouvé. Veuillez vous assurer qu'il est dans le même répertoire que l'application Streamlit.")


