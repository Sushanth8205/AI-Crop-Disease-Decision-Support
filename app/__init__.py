import os
from flask import Flask
from pymongo import MongoClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize MongoDB client
mongo_client = MongoClient(os.getenv('MONGO_URI', 'mongodb://localhost:27017/cropcare'))
db = mongo_client.get_database()

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret')
    
    # Configuration for file uploads
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static/uploads')
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    
    from app.routes.main import main_bp
    from app.routes.auth import auth_bp
    from app.routes.detection import detection_bp
    from app.routes.report import report_bp
    from app.routes.ai import ai_bp
    from app.routes.advisor import advisor_bp
    
    app.register_blueprint(main_bp)
    app.register_blueprint(auth_bp, url_prefix='/auth')
    app.register_blueprint(detection_bp, url_prefix='/detection')
    app.register_blueprint(report_bp, url_prefix='/report')
    app.register_blueprint(ai_bp, url_prefix='/ai')
    app.register_blueprint(advisor_bp, url_prefix='/advisor')
    
    return app
