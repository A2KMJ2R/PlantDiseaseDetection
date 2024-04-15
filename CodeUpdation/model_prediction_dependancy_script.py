# from _future_ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
from werkzeug.utils import secure_filename
# Keras
import tensorflow as tf
from tensorflow.keras.models import load_model


# Load your trained models
plant_type_model = load_model('plant_type_identification_model0.h5')
plant_disease_model = load_model('trained_plant_disease_model0.h5')


def model_predict(img_path, model):
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])

    preds = model.predict(input_arr)
    return preds


def handle_model_predictions():
        
        plant_type_preds = model_predict(file_path, plant_type_model)
        plant_type_pred_class = np.argmax(plant_type_preds)
        plant_type_confidence = np.max(plant_type_preds)

        plant_type_categories = ['APPLE', 'BELLPEPPER', 'CHERRY', 'CORN', 'GRAPE', 'PEACH', 'POTATO', 'STRAWBERRY', 'TOMATO']
        plant_type_result = plant_type_categories[plant_type_pred_class]

        # Print top 5 plant type predictions
        top_plant_type_preds = np.argsort(plant_type_preds)[0][-5:][::-1]
        print("Top 5 Plant Type Predictions:")
        for i, pred in enumerate(top_plant_type_preds):
            print(f"{plant_type_categories[pred]}: {plant_type_preds[0][pred]}")

        if plant_type_confidence > .90:
            # Make prediction for plant disease
            plant_disease_preds = model_predict(file_path, plant_disease_model)
            plant_disease_pred_class = np.argmax(plant_disease_preds)
            plant_disease_confidence = np.max(plant_disease_preds)

            plant_disease_categories = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 
                                        'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
                                        'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
                                        'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy', 
                                        'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy', 
                                        'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 
                                        'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
                                        'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 
                                        'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 
                                        'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
                                         'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
            
            plant_disease_result = plant_disease_categories[plant_disease_pred_class]

            # Print top 5 plant disease predictions
            top_plant_disease_preds = np.argsort(plant_disease_preds)[0][-5:][::-1]
            print("Top 5 Plant Disease Predictions:")
            for i, pred in enumerate(top_plant_disease_preds):
                print(f"{plant_disease_categories[pred]}: {plant_disease_preds[0][pred]}")
            
            if plant_disease_confidence > .90:

                pass
                # code for displaying outputs of prediction

        else:
            plant_type_confidence = 'low'
            plant_disease_confidence = 'Not predicted'
