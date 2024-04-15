from lime import lime_image
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

image_path = r"C:\Users\Justin\Downloads\PlantDisease\test\test1\AppleCedarRust1.JPG"

from lime import lime_image
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Load your pretrained CNN model
model = load_model('plant_disease_identifier_model_m0.h5')

# Wrap your model with the LIME ImageClassifier
explainer = lime_image.LimeImageExplainer()

# Load and preprocess the image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize as needed
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Load and preprocess the image
image = load_and_preprocess_image(image_path)

# Generate explanations for your model's prediction
explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

# Get the explanation data
temp, mask = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=5, hide_rest=False)

# Create an overlay of the image and mask
from skimage.segmentation import mark_boundaries
overlay = mark_boundaries(temp / 2 + 0.5, mask)

# Display the overlay
import matplotlib.pyplot as plt
plt.imshow(overlay)
plt.axis('off')
plt.show()

from lime import lime_image
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# Load your pretrained CNN model
model = load_model('plant_disease_identifier_model_m0.h5')

# Wrap your model with the LIME ImageClassifier
explainer = lime_image.LimeImageExplainer()

# Load and preprocess your image
def load_and_preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize((128, 128))  # Resize as needed
    img = np.array(img) / 255.0  # Normalize pixel values
    return img

# Load and preprocess the image
image = load_and_preprocess_image(image_path)

# Generate explanations for your model's prediction
explanation = explainer.explain_instance(image, model.predict, top_labels=5, hide_color=0, num_samples=1000)

# Get segmentation mask used by Lime
segmentation_mask = explanation.segments

# Get the importance scores associated with each segment
segment_importances = explanation.local_exp[explanation.top_labels[0]]

# Visualize the image with segmentation mask overlaid
plt.imshow(image)
plt.imshow(segmentation_mask, alpha=0.5, cmap='viridis')
plt.title('Image with Lime Segmentation Mask')
plt.colorbar(label='Importance')
plt.show()
