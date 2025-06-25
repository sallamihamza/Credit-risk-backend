import os
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging
from typing import Dict, Any, Tuple
from flask import Blueprint, request, jsonify

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

prediction_bp = Blueprint('prediction', __name__)

class CreditRiskPredictor:
    def __init__(self, pipeline_path: str = None):
        self.pipeline = None
        self.model_info = {}
        self.feature_names = [
            'person_age', 'person_income', 'person_emp_exp', 'loan_amnt',
            'loan_int_rate', 'loan_percent_income', 'cb_person_cred_hist_length',
            'credit_score', 'person_gender', 'person_education',
            'person_home_ownership', 'loan_intent', 'previous_loan_defaults_on_file'
        ]
        if pipeline_path:
            self.load_model(pipeline_path)
        else:
            self.load_model()

    def load_model(self, pipeline_path: str = None) -> bool:
        try:
            if pipeline_path is None:
                pipeline_path = os.path.join(os.path.dirname(__file__), 'models', 'credit_risk_pipeline.pkl')
            self.pipeline = joblib.load(pipeline_path)
            self.model_info = {
                'model_name': self._get_model_name(),
                'model_version': '1.0',
                'features_count': len(self.feature_names),
                'loaded_at': datetime.now().isoformat()
            }
            logger.info(f"Modèle chargé avec succès: {self.model_info['model_name']}")
            return True
        except Exception as e:
            logger.error(f"Erreur lors du chargement du pipeline: {str(e)}")
            return False

    def _get_model_name(self) -> str:
        if hasattr(self.pipeline, 'named_steps') and 'classifier' in self.pipeline.named_steps:
            classifier = self.pipeline.named_steps['classifier']
            return classifier.__class__.__name__
        return type(self.pipeline).__name__

    def validate_input(self, data: Dict[str, Any]) -> Tuple[bool, Dict[str, Any]]:
        errors = {}
        missing_fields = [f for f in self.feature_names if f not in data]
        if missing_fields:
            errors['missing_fields'] = missing_fields

        if 'person_age' in data and (not isinstance(data['person_age'], (int, float)) or not 18 <= data['person_age'] <= 100):
            errors['person_age'] = "L'âge doit être entre 18 et 100 ans"
        if 'person_income' in data and (not isinstance(data['person_income'], (int, float)) or data['person_income'] <= 0):
            errors['person_income'] = "Le revenu doit être positif"
        if 'credit_score' in data and (not isinstance(data['credit_score'], (int, float)) or not 300 <= data['credit_score'] <= 850):
            errors['credit_score'] = "Le score de crédit doit être entre 300 et 850"
        if 'loan_amnt' in data and (not isinstance(data['loan_amnt'], (int, float)) or data['loan_amnt'] <= 0):
            errors['loan_amnt'] = "Le montant du prêt doit être positif"
        if 'loan_int_rate' in data and (not isinstance(data['loan_int_rate'], (int, float)) or not 0 <= data['loan_int_rate'] <= 50):
            errors['loan_int_rate'] = "Le taux d'intérêt doit être entre 0 et 50%"

        categorical_validations = {
            'person_gender': ['Male', 'Female'],
            'person_education': ['High School', 'Bachelor', 'Master', 'Doctorate'],
            'person_home_ownership': ['RENT', 'OWN', 'MORTGAGE', 'OTHER'],
            'loan_intent': ['PERSONAL', 'EDUCATION', 'MEDICAL', 'VENTURE', 'HOMEIMPROVEMENT', 'DEBTCONSOLIDATION'],
            'previous_loan_defaults_on_file': ['No', 'Yes']
        }

        for field, valid in categorical_validations.items():
            if field in data and data[field] not in valid:
                errors[field] = f"Valeur invalide. Valeurs acceptées: {', '.join(valid)}"

        return len(errors) == 0, errors

    def predict(self, data: Dict[str, Any]) -> Dict[str, Any]:
        start_time = datetime.now()
        try:
            if self.pipeline is None:
                raise ValueError("Pipeline non chargé")

            is_valid, validation_errors = self.validate_input(data)
            if not is_valid:
                return {
                    'status': 'error',
                    'error_code': 'VALIDATION_ERROR',
                    'message': "Données d'entrée invalides",
                    'details': validation_errors,
                    'timestamp': datetime.now().isoformat()
                }

            df = pd.DataFrame([data])
            df = df[self.feature_names]
            prediction = self.pipeline.predict(df)[0]
            probability = self.pipeline.predict_proba(df)[0]
            prob_high_risk = probability[1] if len(probability) > 1 else probability[0]
            confidence_level = self._get_confidence_level(prob_high_risk)
            processing_time = (datetime.now() - start_time).total_seconds() * 1000

            return {
                'status': 'success',
                'prediction': {
                    'risk_class': int(prediction),
                    'risk_label': 'Risque élevé' if prediction == 1 else 'Faible risque',
                    'probability_score': round(float(prob_high_risk), 4),
                    'confidence_level': confidence_level
                },
                'model_info': {
                    'model_name': self.model_info.get('model_name', 'Unknown'),
                    'model_version': self.model_info.get('model_version', '1.0'),
                    'features_used': len(self.feature_names)
                },
                'timestamp': datetime.now().isoformat(),
                'processing_time_ms': round(processing_time, 2)
            }

        except Exception as e:
            logger.error(f"Erreur lors de la prédiction: {str(e)}")
            return {
                'status': 'error',
                'error_code': 'PREDICTION_ERROR',
                'message': 'Erreur lors de la prédiction',
                'details': {'error': str(e)},
                'timestamp': datetime.now().isoformat()
            }

    def _get_confidence_level(self, probability: float) -> str:
        if probability >= 0.8 or probability <= 0.2:
            return "Élevé"
        elif probability >= 0.6 or probability <= 0.4:
            return "Moyen"
        else:
            return "Faible"

    def get_model_info(self) -> Dict[str, Any]:
        if self.pipeline is None:
            return {'error': 'Aucun pipeline chargé'}
        return {
            'model_name': self.model_info.get('model_name', 'Unknown'),
            'model_version': self.model_info.get('model_version', '1.0'),
            'features': self.feature_names,
            'features_count': len(self.feature_names),
            'loaded_at': self.model_info.get('loaded_at'),
            'status': 'loaded'
        }

    def health_check(self) -> Dict[str, Any]:
        return {
            'status': 'healthy' if self.pipeline else 'unhealthy',
            'pipeline_loaded': self.pipeline is not None,
            'timestamp': datetime.now().isoformat()
        }

predictor = CreditRiskPredictor()

@prediction_bp.route('/predict', methods=['POST'])
def predict_credit_risk():
    data = request.get_json()
    result = predictor.predict(data)
    return jsonify(result)
