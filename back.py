# app/config.py
import os

class Config:
    UPLOAD_FOLDER = 'static/uploads'
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    MODEL_DIR = 'models'
    
    # Create necessary directories
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

# app/utils.py
import os
from werkzeug.utils import secure_filename
from app.config import Config

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in Config.ALLOWED_EXTENSIONS

def save_upload_file(file):
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(Config.UPLOAD_FOLDER, filename)
        file.save(filepath)
        return filepath, filename
    return None, None

# app/prediction_service.py
import numpy as np
import tensorflow as tf
import cv2
import pickle
import os
from app.config import Config

class PredictionService:
    def __init__(self):
        self.img_size = 64
        self.models_loaded = False
        
    def load_models(self):
        """Load all necessary models and encoders"""
        if not self.models_loaded:
            try:
                self.gender_model = tf.keras.models.load_model(
                    os.path.join(Config.MODEL_DIR, 'gender_model.h5'))
                self.male_age_model = tf.keras.models.load_model(
                    os.path.join(Config.MODEL_DIR, 'male_age_model.h5'))
                self.female_age_model = tf.keras.models.load_model(
                    os.path.join(Config.MODEL_DIR, 'female_age_model.h5'))
                
                with open(os.path.join(Config.MODEL_DIR, 'gender_encoder.pkl'), 'rb') as f:
                    self.gender_encoder = pickle.load(f)
                with open(os.path.join(Config.MODEL_DIR, 'age_encoder.pkl'), 'rb') as f:
                    self.age_encoder = pickle.load(f)
                
                self.models_loaded = True
                print("Models loaded successfully!")
            except Exception as e:
                print(f"Error loading models: {str(e)}")
                raise
    
    def preprocess_image(self, image_path):
        """Preprocess image for prediction"""
        try:
            img = cv2.imread(image_path)
            if img is None:
                raise ValueError("Could not read image file")
            
            img = cv2.resize(img, (self.img_size, self.img_size))
            img = img / 255.0
            return np.expand_dims(img, axis=0)
        except Exception as e:
            print(f"Error preprocessing image: {str(e)}")
            raise
    
    def predict(self, image_path):
        """Make predictions for a given image"""
        if not self.models_loaded:
            self.load_models()
        
        try:
            # Preprocess image
            img = self.preprocess_image(image_path)
            
            # Predict gender
            gender_pred = self.gender_model.predict(img, verbose=0)
            gender = self.gender_encoder.inverse_transform([round(gender_pred[0][0])])[0]
            
            # Predict age based on gender
            if gender == 'male':
                age_pred = self.male_age_model.predict(img, verbose=0)
            else:
                age_pred = self.female_age_model.predict(img, verbose=0)
            
            age_class = np.argmax(age_pred)
            age_range = self.age_encoder.inverse_transform([age_class])[0]
            
            confidence_scores = {
                'gender_confidence': float(abs(0.5 - gender_pred[0][0]) * 2),
                'age_confidence': float(np.max(age_pred))
            }
            
            return {
                'gender': gender,
                'age_range': age_range,
                'confidence_scores': confidence_scores
            }
            
        except Exception as e:
            print(f"Error during prediction: {str(e)}")
            raise

# app/routes.py
from flask import Blueprint, request, jsonify, render_template
from app.utils import save_upload_file
from app.prediction_service import PredictionService

main = Blueprint('main', __name__)
predictor = PredictionService()

@main.route('/')
def home():
    return render_template('index.html')

@main.route('/predict', methods=['POST'])
def predict():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        filepath, filename = save_upload_file(file)
        if not filepath:
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Make prediction
        result = predictor.predict(filepath)
        
        # Add image path to result
        result['image_path'] = f'uploads/{filename}'
        
        return jsonify({
            'success': True,
            **result
        })
        
    except Exception as e:
        print(f"Error in /predict endpoint: {str(e)}")
from flask import Flask
from app.routes import main

app = Flask(__name__)
app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)