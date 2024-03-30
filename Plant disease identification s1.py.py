import cv2
import matplotlib.pyplot as plt
import tensorflow as tf
import pandas as pd
import os
from sklearn.model_selection import train_test_split

# Define paths
data_dir = "/content/gdrive/MyDrive/Plant_project/plant_dis/Leaf-Classification-master/data/"
images_dir = os.path.join(data_dir, "images")
train_csv_path = os.path.join(data_dir, "train.csv")

# Read train data
train_df = pd.read_csv(train_csv_path)

# Split data into train and validation sets
train_data, val_data = train_test_split(train_df, test_size=0.2, random_state=42)

# Data augmentation and preprocessing
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_df['id'] = train_df['id'].astype(str)
test_df['id'] = test_df['id'].astype(str)

train_df['species'] = train_df['species'].astype(str)
test_df['species'] = test_df['species'].astype(str)

# Create data generators
train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_data,
    directory=images_dir,
    x_col="id",
    y_col="species",
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical'
)

val_generator = val_datagen.flow_from_dataframe(
    dataframe=val_data,
    directory=images_dir,
    x_col="id",
    y_col="species",
    target_size=(512, 512),
    batch_size=64,
    class_mode='categorical'
)

# Define the model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(512, 512, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2, 2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(6, activation='softmax')  # Adjust number of classes accordingly
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,  # Adjust number of epochs as needed
    validation_data=val_generator
)

# Evaluate the model
test_loss, test_acc = model.evaluate(val_generator, verbose=2)
print('\nValidation accuracy:', test_acc)

# Plot training and validation accuracy
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc=0)
plt.show()
