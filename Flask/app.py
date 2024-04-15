# from _future_ import division, print_function
import sys
import os
import glob
import re
import numpy as np
import cv2
# Flask utils
from flask import Flask, redirect, url_for, request, render_template, jsonify
from werkzeug.utils import secure_filename
# Keras
import tensorflow as tf
from tensorflow.keras.models import load_model

# # Define a flask app
# app = Flask(__name__)

# # Model saved with Keras model.save()
# # MODEL_PATH = 'plant_type_identification_model1.keras'
# MODEL_PATH = 'trained_plant_disease_model0.h5'

# # Load your trained model
# model = load_model(MODEL_PATH)

# print('Model loaded. Check http://127.0.0.1:5000/')


# def model_predict(img_path, model):
#     # Update by ViPS
#     # img = cv2.imread(img_path)
#     # new_arr = cv2.resize(img, (100, 100))
#     # new_arr = np.array(new_arr / 255)
#     # new_arr = new_arr.reshape(-1, 100, 100, 3)
#     image = tf.keras.preprocessing.image.load_img(img_path,target_size=(128,128))
#     input_arr = tf.keras.preprocessing.image.img_to_array(image)
#     input_arr = np.array([input_arr])
#     new_arr = input_arr

#     preds = model.predict(new_arr)
#     return preds


# @app.route('/', methods=['GET', 'POST'])
# def index():
#     if request.method == 'POST':
#         # Get the file from post request
#         f = request.files['file']

#         # Save the file to ./uploads
#         basepath = os.path.dirname(__file__)
#         file_path = os.path.join(
#             basepath, 'uploads', secure_filename(f.filename))
#         f.save(file_path)

#         # Make prediction
#         preds = model_predict(file_path, model)

#         # Process your result for human
#         pred_class = preds.argmax()  # Simple argmax

#         CATEGORIES1 = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
#                       'Blueberry_healthy', 'Cherry(including_sour)_healthy', 'Cherry(including_sour)_Powdery_mildew',
#                       'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn(maize)__healthy',
#                       'Corn(maize)_Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_healthy',
#                       'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange_Haunglongbing(Citrus_greening)', 'Peach_Bacterial_spot',
#                       'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_healthy',
#                       'Potato_Late_blight', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_healthy',
#                       'Strawberry_Leaf_scorch', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
#                       'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
#                       'Tomato_Tomato_mosaic_virus', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus']
#         CATEGORIES2 = ['APPLE', 'BELLPEPPER', 'CHERRY', 'CORN', 'GRAPE', 'PEACH', 'POTATO', 'STRAWBERRY', 'TOMATO']
        
#         return CATEGORIES1[pred_class]

#     return render_template('index.html')


# if __name__ == '__main__':
#     app.run(debug=True)

##############################################################################################
# Define a flask app
app = Flask(__name__)

# Load your trained models
plant_type_model = load_model('plant_type_identification_model0.h5')
plant_disease_model = load_model('trained_plant_disease_model0.h5')

print('Models loaded. Check http://127.0.0.1:5000/')


def model_predict(img_path, model):
    # Update by ViPS
    image = tf.keras.preprocessing.image.load_img(img_path, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])
    new_arr = input_arr

    preds = model.predict(new_arr)
    return preds


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction for plant type
        plant_type_preds = model_predict(file_path, plant_type_model)
        plant_type_pred_class = np.argmax(plant_type_preds)
        plant_type_confidence = np.max(plant_type_preds)  # Get the maximum confidence
        plant_type_categories = ['APPLE', 'BELLPEPPER', 'CHERRY', 'CORN', 'GRAPE', 'PEACH', 'POTATO', 'STRAWBERRY', 'TOMATO']
        plant_type_result = plant_type_categories[plant_type_pred_class]

        # Make prediction for plant disease
        plant_disease_preds = model_predict(file_path, plant_disease_model)
        plant_disease_pred_class = np.argmax(plant_disease_preds)
        plant_disease_confidence = np.max(plant_disease_preds)  # Get the maximum confidence
        plant_disease_categories = ['Apple__Apple_scab', 'Apple_Black_rot', 'Apple_Cedar_apple_rust', 'Apple_healthy',
                                     'Blueberry_healthy', 'Cherry(including_sour)_healthy', 'Cherry(including_sour)_Powdery_mildew',
                                     'Corn(maize)_Cercospora_leaf_spot Gray_leaf_spot', 'Corn(maize)Common_rust', 'Corn(maize)__healthy',
                                     'Corn(maize)_Northern_Leaf_Blight', 'Grape_Black_rot', 'Grape_Esca(Black_Measles)', 'Grape_healthy',
                                     'Grape_Leaf_blight(Isariopsis_Leaf_Spot)', 'Orange_Haunglongbing(Citrus_greening)', 'Peach_Bacterial_spot',
                                     'Peach_healthy', 'Pepper,_bell_Bacterial_spot', 'Pepper,_bell_healthy', 'Potato_Early_blight', 'Potato_healthy',
                                     'Potato_Late_blight', 'Raspberry_healthy', 'Soybean_healthy', 'Squash_Powdery_mildew', 'Strawberry_healthy',
                                     'Strawberry_Leaf_scorch', 'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_healthy', 'Tomato_Late_blight',
                                     'Tomato_Leaf_Mold', 'Tomato_Septoria_leaf_spot', 'Tomato_Spider_mites Two-spotted_spider_mite', 'Tomato_Target_Spot',
                                     'Tomato_Tomato_mosaic_virus', 'Tomato__Tomato_Yellow_Leaf_Curl_Virus']
        plant_disease_result = plant_disease_categories[plant_disease_pred_class]

        # Return results including confidence
        return jsonify({"plant_type": plant_type_result, "plant_disease": plant_disease_result,
                        "plant_type_confidence": float(plant_type_confidence),
                        "plant_disease_confidence": float(plant_disease_confidence),
                        "image_path": file_path})

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)