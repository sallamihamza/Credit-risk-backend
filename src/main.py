import os
import sys
import logging
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Ne pas changer cette ligne
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from src.models.user import db
from src.routes.prediction import prediction_bp
from src.prediction_service import predictor

# Initialisation de Flask
app = Flask(
    __name__,
    static_folder=os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(__file__))),
        'credit-risk-frontend',
        'dist'
    )
)
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# CORS
CORS(app, origins="*")

# Base de données (optionnelle)
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{os.path.join(os.path.dirname(__file__), 'database', 'app.db')}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

with app.app_context():
    db.init_app(app)
    db.create_all()

# Routes
app.register_blueprint(prediction_bp, url_prefix='/api/v1')

@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    static_folder_path = app.static_folder
    if path and os.path.exists(os.path.join(static_folder_path, path)):
        return send_from_directory(static_folder_path, path)
    index_path = os.path.join(static_folder_path, 'index.html')
    if os.path.exists(index_path):
        return send_from_directory(static_folder_path, 'index.html')
    return jsonify({
        'message': 'API de Prédiction de Risque de Crédit',
        'version': '1.0.0',
        'endpoints': {
            'predict': '/api/v1/predict',
            'health': '/api/v1/health',
            'model_info': '/api/v1/model/info',
            'features': '/api/v1/features',
            'example': '/api/v1/example'
        },
        'documentation': 'Consultez /api/v1/example pour un exemple de requête'
    })

@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'status': 'error',
        'error_code': 'NOT_FOUND',
        'message': 'Endpoint non trouvé',
        'available_endpoints': [
            '/api/v1/predict',
            '/api/v1/health',
            '/api/v1/model/info',
            '/api/v1/features',
            '/api/v1/example'
        ]
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'status': 'error',
        'error_code': 'INTERNAL_ERROR',
        'message': 'Erreur interne du serveur'
    }), 500

def initialize_model():
    try:
        pipeline_path = os.path.join(os.path.dirname(__file__), 'models', 'credit_risk_pipeline.pkl')
        if os.path.exists(pipeline_path):
            success = predictor.load_model(pipeline_path)
            if success:
                logger.info("Modèle chargé avec succès au démarrage")
            else:
                logger.warning("Échec du chargement du pipeline")
        else:
            logger.warning(f"Pipeline non trouvé à l'emplacement: {pipeline_path}")
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")

if __name__ == '__main__':
    initialize_model()
    logger.info("Démarrage de l'API de Prédiction de Risque de Crédit")
    app.run(host='0.0.0.0', port=5000, debug=True)
