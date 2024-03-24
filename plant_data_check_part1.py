# -*- coding: utf-8 -*-
"""Plant_data_check_part1

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xbdEOoru7Cm1YVEt3AGItql6fk3wmuaD
"""

!git clone https://github.com/mirkousuelli/CNN-Leaves-Classifier

!ls CNN-Leaves-Classifier/

import cv2
import matplotlib.pyplot as plt

import tensorflow as tf



from google.colab import drive
drive.mount('/content/gdrive')

dataset = tf.keras.preprocessing.image_dataset_from_directory("/content/gdrive/MyDrive/Plant_project/train")
ds_train = tf.keras.preprocessing.image_dataset_from_directory("/content/gdrive/MyDrive/Plant_project/train", validation_split = 0.2, subset = "training", seed = 123)
ds_validation = tf.keras.preprocessing.image_dataset_from_directory("/content/gdrive/MyDrive/Plant_project/test", validation_split = 0.2, subset = "validation", seed = 123)

#Define batch size and print images
import tensorflow_datasets as tfds

batch_size = 64
dataset_names = dataset
class_names = dataset.class_names
print(class_names)

#standardize data for CNN
size = (512, 512)
ds_train = ds_train.map(lambda image, label: (tf.image.resize(image, size), label))
ds_val = ds_validation.map(lambda image, label: (tf.image.resize(image, size),label))

#Display 9 images
import matplotlib.pyplot as plt

plt.figure(figsize=(10,10))
for images, labels in ds_train.take(1):
  for i in range(9):
    ax = plt.subplot(3,3,i+1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")

#deep processing
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers

image = Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(height_factor=(-0.2, -0.3), width_factor=(-0.2, -0.3), interpolation = 'bilinear'),
        layers.RandomContrast(factor=0.1),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
    ],
    name= "image",)

import numpy as np

for images, labels in ds_train.take(1):
  plt.figure(figsize=(10,10))
  first_image = images[0]
  def f(x):
    return int(x)
  f2 = np.vectorize(f)
  for i in range(9):
    ax= plt.subplot(3,3,i+1)
    augmented_image = image(
        tf.expand_dims(first_image, 0), training = True
        )
    plt.imshow(augmented_image[0].numpy().astype("int32"))
    plt.title(f2(labels[0]))
    plt.axis("off")

