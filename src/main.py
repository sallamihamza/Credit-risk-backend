import os
import sys
import logging
from flask import Flask, send_from_directory, jsonify
from flask_cors import CORS

# Ajouter les handlers Gunicorn pour Railway
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
gunicorn_logger = logging.getLogger('gunicorn.error')
if gunicorn_logger.handlers:
    logger.handlers = gunicorn_logger.handlers

# Fix path issues for Railway deployment
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)

# Only add to path if not already there
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import after path setup
try:
    from src.models.user import db
    from src.routes.prediction import prediction_bp
    from src.prediction_service import predictor
except ImportError as e:
    logger.error(f"Import error: {e}")
    # Create minimal imports for testing
    db = None
    prediction_bp = None
    predictor = None

# Initialisation de l'application Flask
# Fix static folder path for Railway
static_folder = os.path.join(project_root, 'static') if os.path.exists(os.path.join(project_root, 'static')) else None

app = Flask(__name__, static_folder=static_folder)
app.config['SECRET_KEY'] = 'asdf#FGSgvasgf$5$WGT'

# Configuration CORS pour autoriser GitHub Pages
CORS(app, origins=["https://sallamihamza.github.io", "http://localhost:5173"])

# Configuration de la base de données SQLite - use /tmp for Railway
db_dir = '/tmp' if os.path.exists('/tmp') else current_dir
db_path = os.path.join(db_dir, 'app.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f"sqlite:///{db_path}"
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

logger.info(f"Database path: {db_path}")
logger.info(f"Static folder: {static_folder}")

# Initialisation de la base de données
if db:
    try:
        with app.app_context():
            db.init_app(app)
            db.create_all()
        logger.info("Database initialized successfully")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

# Enregistrement du blueprint pour les routes API
if prediction_bp:
    app.register_blueprint(prediction_bp, url_prefix='/api/v1')
else:
    logger.warning("Prediction blueprint not available")

# Route principale (frontend SPA ou fallback)
@app.route('/', defaults={'path': ''})
@app.route('/<path:path>')
def serve(path):
    if app.static_folder and path and os.path.exists(os.path.join(app.static_folder, path)):
        return send_from_directory(app.static_folder, path)
    elif app.static_folder and os.path.exists(os.path.join(app.static_folder, 'index.html')):
        return send_from_directory(app.static_folder, 'index.html')
    else:
        return jsonify({
            'message': 'API de Prédiction de Risque de Crédit',
            'version': '1.0.0',
            'status': 'running',
            'environment': 'production' if os.environ.get('RAILWAY_ENVIRONMENT') else 'development',
            'endpoints': {
                'predict': '/api/v1/predict',
                'health': '/api/v1/health',
                'model_info': '/api/v1/model/info',
                'features': '/api/v1/features',
                'example': '/api/v1/example'
            },
            'documentation': 'Consultez /api/v1/example pour un exemple de requête'
        })

# Health check endpoint
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'database': 'connected' if db else 'not available',
        'model': 'loaded' if predictor and hasattr(predictor, 'pipeline') else 'not loaded'
    })

# Gestion des erreurs 404
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
            '/api/v1/example',
            '/health'
        ]
    }), 404

# Gestion des erreurs 500
@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal error: {error}")
    return jsonify({
        'status': 'error',
        'error_code': 'INTERNAL_ERROR',
        'message': 'Erreur interne du serveur'
    }), 500

# Chargement du modèle ML
def initialize_model():
    if not predictor:
        logger.warning("Predictor not available, skipping model initialization")
        return
        
    try:
        # Try multiple possible paths for the model
        possible_paths = [
            os.path.join(current_dir, 'models', 'credit_risk_pipeline.pkl'),
            os.path.join(project_root, 'src', 'models', 'credit_risk_pipeline.pkl'),
            os.path.join(project_root, 'models', 'credit_risk_pipeline.pkl')
        ]
        
        for pipeline_path in possible_paths:
            if os.path.exists(pipeline_path):
                success = predictor.load_model(pipeline_path)
                if success:
                    logger.info(f"Modèle chargé avec succès depuis: {pipeline_path}")
                    return
                else:
                    logger.warning(f"Échec du chargement depuis: {pipeline_path}")
        
        logger.warning("Aucun modèle trouvé dans les emplacements suivants:")
        for path in possible_paths:
            logger.warning(f"  - {path}")
            
    except Exception as e:
        logger.error(f"Erreur lors de l'initialisation du modèle: {str(e)}")

# Initialisation du modèle au démarrage
initialize_model()

# For development
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))