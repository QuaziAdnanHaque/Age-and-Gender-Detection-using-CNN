from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import tensorflow as tf

app = Flask(__name__)
CORS(app)

# Load models and encoders
gender_model = tf.keras.models.load_model('gender_model.h5')
male_age_model = tf.keras.models.load_model('male_age_model.h5')
female_age_model = tf.keras.models.load_model('female_age_model.h5')
gender_labels = ['female', 'male']
age_ranges = ['0-12', '13-19', '20-30', '31-45', '46-60', '60+']

def preprocess_image(image):
    img = cv2.resize(image, (64, 64))
    img = img / 255.0
    return np.expand_dims(img, axis=0)

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    npimg = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    processed_image = preprocess_image(image)

    # Predict gender
    gender_pred = gender_model.predict(processed_image)[0][0]
    gender = gender_labels[round(gender_pred)]

    # Predict age based on gender
    if gender == 'male':
        age_pred = male_age_model.predict(processed_image)
    else:
        age_pred = female_age_model.predict(processed_image)
    age_range = age_ranges[np.argmax(age_pred)]

    return jsonify({'gender': gender, 'age_range': age_range})

if __name__ == '__main__':
    app.run(debug=True)
