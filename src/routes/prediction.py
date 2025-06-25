from flask import Blueprint, request, jsonify
from src.prediction_service import predictor
import logging

logger = logging.getLogger(__name__)
prediction_bp = Blueprint('prediction', __name__)

@prediction_bp.route('/predict', methods=['POST'])
def predict_credit_risk():
    if not request.is_json:
        return jsonify({
            'status': 'error',
            'error_code': 'INVALID_CONTENT_TYPE',
            'message': 'Content-Type doit être application/json'
        }), 400

    data = request.get_json()
    if not data:
        return jsonify({
            'status': 'error',
            'error_code': 'EMPTY_REQUEST',
            'message': 'Corps de requête vide'
        }), 400

    try:
        result = predictor.predict(data)
        status_code = 200 if result['status'] == 'success' else 400
        return jsonify(result), status_code
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint /predict: {str(e)}")
        return jsonify({
            'status': 'error',
            'error_code': 'INTERNAL_ERROR',
            'message': 'Erreur interne du serveur',
            'details': {'error': str(e)}
        }), 500

@prediction_bp.route('/health', methods=['GET'])
def health_check():
    try:
        health_status = predictor.health_check()
        health_status.update({
            'version': '1.0.0',
            'api_status': 'running'
        })
        status_code = 200 if health_status['status'] == 'healthy' else 503
        return jsonify(health_status), status_code
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint /health: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Erreur lors de la vérification de santé',
            'details': {'error': str(e)}
        }), 500

@prediction_bp.route('/model/info', methods=['GET'])
def get_model_info():
    try:
        model_info = predictor.get_model_info()
        if 'error' in model_info:
            return jsonify({
                'status': 'error',
                'message': model_info['error']
            }), 503
        return jsonify(model_info), 200
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint /model/info: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Erreur lors de la récupération des informations du modèle',
            'details': {'error': str(e)}
        }), 500

@prediction_bp.route('/features', methods=['GET'])
def get_features():
    try:
        features_info = {
            'features': predictor.feature_names,
            'count': len(predictor.feature_names),
            'description': {
                'person_age': 'Âge (18-100)',
                'person_income': 'Revenu annuel',
                'person_emp_exp': 'Années d\'expérience',
                'loan_amnt': 'Montant du prêt',
                'loan_int_rate': 'Taux d\'intérêt (%)',
                'loan_percent_income': 'Pourcentage revenu/prêt',
                'cb_person_cred_hist_length': 'Historique crédit',
                'credit_score': 'Score (300-850)',
                'person_gender': 'Genre',
                'person_education': 'Niveau d\'étude',
                'person_home_ownership': 'Statut logement',
                'loan_intent': 'Objet du prêt',
                'previous_loan_defaults_on_file': 'Défauts précédents'
            }
        }
        return jsonify(features_info), 200
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint /features: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Erreur lors de la récupération des features',
            'details': {'error': str(e)}
        }), 500

@prediction_bp.route('/example', methods=['GET'])
def get_example_request():
    try:
        example = {
            'description': 'Exemple de requête',
            'endpoint': '/api/v1/predict',
            'method': 'POST',
            'headers': {
                'Content-Type': 'application/json'
            },
            'body': {
                "person_age": 30,
                "person_income": 50000,
                "person_emp_exp": 5,
                "loan_amnt": 15000,
                "loan_int_rate": 10.5,
                "loan_percent_income": 0.3,
                "cb_person_cred_hist_length": 7,
                "credit_score": 680,
                "person_gender": "Male",
                "person_education": "Bachelor",
                "person_home_ownership": "RENT",
                "loan_intent": "PERSONAL",
                "previous_loan_defaults_on_file": "No"
            },
            'expected_response': {
                'status': 'success',
                'prediction': {
                    'risk_class': 0,
                    'risk_label': 'Faible risque',
                    'probability_score': 0.23,
                    'confidence_level': 'Élevé'
                },
                'model_info': {
                    'model_name': 'RandomForestClassifier',
                    'model_version': '1.0',
                    'features_used': 13
                }
            }
        }
        return jsonify(example), 200
    except Exception as e:
        logger.error(f"Erreur dans l'endpoint /example: {str(e)}")
        return jsonify({
            'status': 'error',
            'message': 'Erreur lors de la génération de l\'exemple',
            'details': {'error': str(e)}
        }), 500
