import tkinter as tk
from tkinter import filedialog
import cv2
from PIL import Image, ImageTk

def select_image():
    path = filedialog.askopenfilename()
    if len(path) > 0:
        detect_disease(path)

def detect_disease(image_path):
    # Your disease detection code using OpenCV or any other library
    # For demonstration, let's just display the image
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(image)
    image.thumbnail((300, 300))
    photo = ImageTk.PhotoImage(image)

    label_image.config(image=photo)
    label_image.image = photo

# Create GUI
root = tk.Tk()
root.title("Plant Disease Detection")

btn_select = tk.Button(root, text="Select Image", command=select_image)
btn_select.pack(padx=10, pady=10)

label_image = tk.Label(root)
label_image.pack(padx=10, pady=10)

root.mainloop()
