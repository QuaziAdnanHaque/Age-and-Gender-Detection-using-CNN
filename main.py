import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
import cv2
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# Define dataset paths
FEMALE_PATH = "F:/project/utkface_aligned_cropped/female"
MALE_PATH = "F:/project/utkface_aligned_cropped/male"
ALL_PHOTOS_PATH = "F:/project/UTKFace"

class AgeGenderRecognition:
    def __init__(self):
        self.age_ranges = ['0-12', '13-19', '20-30', '31-45', '46-60', '60+']
        self.img_size = 64
        
    def load_and_preprocess_utkface(self):
        """
        Load and preprocess images from UTKFace dataset
        File format: [age]_[gender]_[race]_[date&time].jpg
        """
        images = []
        genders = []
        ages = []
        
        # Process all photos
        for img_name in os.listdir(ALL_PHOTOS_PATH):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                try:
                    # Parse filename
                    age, gender = img_name.split('_')[:2]
                    age = int(age)
                    gender = 'male' if int(gender) == 1 else 'female'
                    
                    # Load and preprocess image
                    img_path = os.path.join(ALL_PHOTOS_PATH, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img / 255.0  # Normalize
                        
                        images.append(img)
                        genders.append(gender)
                        ages.append(self.categorize_age(age))
                except Exception as e:
                    print(f"Error processing {img_name}: {str(e)}")
                    continue
        
        return np.array(images), np.array(genders), np.array(ages)
    
    def load_gender_specific_data(self):
        """
        Load gender-specific datasets
        """
        male_images = []
        female_images = []
        male_ages = []
        female_ages = []
        
        # Process male images
        for img_name in os.listdir(MALE_PATH):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                try:
                    age = int(img_name.split('_')[0])
                    img_path = os.path.join(MALE_PATH, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img / 255.0
                        male_images.append(img)
                        male_ages.append(self.categorize_age(age))
                except:
                    continue
        
        # Process female images
        for img_name in os.listdir(FEMALE_PATH):
            if img_name.endswith('.jpg') or img_name.endswith('.png'):
                try:
                    age = int(img_name.split('_')[0])
                    img_path = os.path.join(FEMALE_PATH, img_name)
                    img = cv2.imread(img_path)
                    if img is not None:
                        img = cv2.resize(img, (self.img_size, self.img_size))
                        img = img / 255.0
                        female_images.append(img)
                        female_ages.append(self.categorize_age(age))
                except:
                    continue
                    
        return (np.array(male_images), np.array(male_ages), 
                np.array(female_images), np.array(female_ages))
    
    def categorize_age(self, age):
        """
        Categorize age into predefined ranges
        """
        if age <= 12: return '0-12'
        elif age <= 19: return '13-19'
        elif age <= 30: return '20-30'
        elif age <= 45: return '31-45'
        elif age <= 60: return '46-60'
        else: return '60+'
    
    def build_gender_model(self):
        """
        Build CNN model for gender classification
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(1, activation='sigmoid')
        ])
        return model
    
    def build_age_model(self, gender):
        """
        Build CNN model for age classification for specific gender
        """
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(self.img_size, self.img_size, 3)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation='relu'),
            layers.Dropout(0.5),
            layers.Dense(len(self.age_ranges), activation='softmax')
        ])
        return model
    
    def train_models(self):
        """
        Train separate models for gender and age classification
        """
        print("Loading and preprocessing UTKFace dataset...")
        X, y_gender, y_age = self.load_and_preprocess_utkface()
        
        # Split data for gender model
        X_train, X_test, gender_train, gender_test, age_train, age_test = train_test_split(
            X, y_gender, y_age, test_size=0.2, random_state=42
        )
        
        # Encode labels
        gender_encoder = LabelEncoder()
        gender_train_encoded = gender_encoder.fit_transform(gender_train)
        gender_test_encoded = gender_encoder.transform(gender_test)
        
        print("Training gender model...")
        gender_model = self.build_gender_model()
        gender_model.compile(optimizer='adam',
                           loss='binary_crossentropy',
                           metrics=['accuracy'])
        
        gender_model.fit(X_train, gender_train_encoded,
                        epochs=20,
                        validation_data=(X_test, gender_test_encoded),
                        batch_size=32)
        
        print("Loading gender-specific datasets...")
        male_images, male_ages, female_images, female_ages = self.load_gender_specific_data()
        
        # Encode age labels
        age_encoder = LabelEncoder()
        male_ages_encoded = age_encoder.fit_transform(male_ages)
        female_ages_encoded = age_encoder.transform(female_ages)
        
        print("Training male age model...")
        male_age_model = self.build_age_model('male')
        male_age_model.compile(optimizer='adam',
                             loss='sparse_categorical_crossentropy',
                             metrics=['accuracy'])
        
        male_age_model.fit(male_images,
                          male_ages_encoded,
                          epochs=20,
                          validation_split=0.2,
                          batch_size=32)
        
        print("Training female age model...")
        female_age_model = self.build_age_model('female')
        female_age_model.compile(optimizer='adam',
                               loss='sparse_categorical_crossentropy',
                               metrics=['accuracy'])
        
        female_age_model.fit(female_images,
                            female_ages_encoded,
                            epochs=20,
                            validation_split=0.2,
                            batch_size=32)
        
        return gender_model, male_age_model, female_age_model, gender_encoder, age_encoder
    
    def predict(self, image_path, gender_model, male_age_model, female_age_model, 
                gender_encoder, age_encoder):
        """
        Make predictions on new images
        """
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (self.img_size, self.img_size))
        img = img / 255.0
        img = np.expand_dims(img, axis=0)
        
        # Predict gender
        gender_pred = gender_model.predict(img)
        gender = gender_encoder.inverse_transform([round(gender_pred[0][0])])[0]
        
        # Predict age based on gender
        if gender == 'male':
            age_pred = male_age_model.predict(img)
        else:
            age_pred = female_age_model.predict(img)
        
        age_class = np.argmax(age_pred)
        age_range = age_encoder.inverse_transform([age_class])[0]
        
        return gender, age_range

# Usage example
if __name__ == "__main__":
    # Initialize the recognition system
    recognition = AgeGenderRecognition()
    
    # Train models
    print("Starting training process...")
    gender_model, male_age_model, female_age_model, gender_encoder, age_encoder = \
        recognition.train_models()
    
    # Save models
    print("Saving models...")
    gender_model.save('gender_model.h5')
    male_age_model.save('male_age_model.h5')
    female_age_model.save('female_age_model.h5')
    
    # Test the model with a sample image
    test_image_path = os.path.join(ALL_PHOTOS_PATH, os.listdir(ALL_PHOTOS_PATH)[0])
    gender, age_range = recognition.predict(test_image_path,
                                          gender_model,
                                          male_age_model,
                                          female_age_model,
                                          gender_encoder,
                                          age_encoder)
    
    print(f"Predicted Gender: {gender}")
    print(f"Predicted Age Range: {age_range}")