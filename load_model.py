import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the saved model
loaded_model = tf.keras.models.load_model("my_model.h5")

# Load and preprocess the new image
img = image.load_img("appleleaves/crn.jpg", target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
img_array = img_array / 255.0  # Normalize pixel values

# Make predictions
predictions = loaded_model.predict(img_array)

# Get the predicted class label
predicted_class = np.argmax(predictions)

print("Predicted class label:", predicted_class)

#edited
 