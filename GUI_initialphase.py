import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import tensorflow as tf
import numpy as np

# Load the trained model
model = tf.keras.models.load_model('trainedmodel.h5')

# Define class names
class_names = ['apple', 'strawberry', 'tomato']

# Function to load and preprocess the image
def load_and_preprocess_image(image_path):
    # Load the image
    img = Image.open(image_path)
    # Resize the image to the required input size of the model
    img = img.resize((224, 224))
    # Convert the image to numpy array
    img_array = np.array(img) / 255.0  # Normalize pixel values
    # Add batch dimension
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Function to perform prediction
def predict_image():
    # Get the path of the selected image file
    file_path = filedialog.askopenfilename()
    # Load and preprocess the image
    img_array = load_and_preprocess_image(file_path)
    # Perform prediction
    predictions = model.predict(img_array)
    # Get the predicted class index
    predicted_class_index = np.argmax(predictions[0])
    # Get the predicted class name
    predicted_class_name = class_names[predicted_class_index]
    # Update the label text with the predicted class name
    result_label.config(text=f"Predicted class: {predicted_class_name}")
    # Open the image using PIL and display it in the GUI
    img = Image.open(file_path)
    img = img.resize((300, 300))  # Resize the image for display
    img = ImageTk.PhotoImage(img)
    img_label.config(image=img)
    img_label.image = img  # Keep a reference to avoid garbage collection

# Create the main application window
root = tk.Tk()
root.title("Plant Disease Detector")

# Set the background color to light blue
root.config(bg="lightblue")

# Create a label for the title
title_label = tk.Label(root, text="Plant Disease Detector", font=("Helvetica", 16, "bold"), bg="lightblue")
title_label.grid(row=0, column=1, pady=10)

# Create a button to select an image with an outline
select_button = tk.Button(root, text="Select Image", command=predict_image, bg="white", fg="black", highlightthickness=2)
select_button.grid(row=1, column=0, padx=20, pady=10)

# Create a label to display the selected image with an outline
img_label = tk.Label(root, bg="lightblue", highlightthickness=2)
img_label.grid(row=1, column=2, padx=20, pady=10)

# Create a label to display the predicted class with an outline
result_label = tk.Label(root, text="", bg="lightblue", highlightthickness=2)
result_label.grid(row=2, column=1, pady=10)

# Set the size of the main window
root.geometry("600x400")

# Run the main event loop
root.mainloop()
