import os
import cv2

def apply_gaussian_blur_to_folder(folder_path, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get list of subfolders in the main folder
    subfolders = [f.path for f in os.scandir(folder_path) if f.is_dir()]

    for subfolder in subfolders:
        # Create corresponding subfolder in the output directory
        output_subfolder = os.path.join(output_folder, os.path.basename(subfolder))
        if not os.path.exists(output_subfolder):
            os.makedirs(output_subfolder)

        # Apply Gaussian blur to each image in the subfolder
        for file in os.listdir(subfolder):
            if file.endswith(".JPG"):
                image_path = os.path.join(subfolder, file)
                img = cv2.imread(image_path)
                blurred_img = cv2.GaussianBlur(img, (5, 5), 0)  # Adjust kernel size as needed
                output_path = os.path.join(output_subfolder, file)
                cv2.imwrite(output_path, blurred_img)

#My folders
train_folder = "train"
test_folder = "test"
valid_folder = "valid"

# Define output folders for blurred images
train_blurred_folder = "n_train"
test_blurred_folder = "n_test"
valid_blurred_folder = "n_valid"

Apply Gaussian blur to images in the train, test, and valid folders
apply_gaussian_blur_to_folder(train_folder, train_blurred_folder)
apply_gaussian_blur_to_folder(test_folder, test_blurred_folder)
apply_gaussian_blur_to_folder(valid_folder, valid_blurred_folder)
