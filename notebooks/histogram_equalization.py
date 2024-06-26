import os
import cv2

# Function to apply histogram equalization to all JPG files in a folder and save them to a new folder
def apply_histogram_equalization(input_folder, output_folder):
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(".jpg"):
                input_file_path = os.path.join(root, file)
                output_file_path = os.path.join(output_folder, os.path.relpath(root, input_folder), file)
                
                # Create output folder if it doesn't exist
                os.makedirs(os.path.dirname(output_file_path), exist_ok=True)
                
                # Read image in grayscale
                image = cv2.imread(input_file_path, cv2.IMREAD_GRAYSCALE)
                if image is not None:
                    # Apply histogram equalization
                    equalized_image = cv2.equalizeHist(image)
                    # Save processed image to output folder
                    cv2.imwrite(output_file_path, equalized_image)
                    print(f"Histogram equalization applied to: {output_file_path}")

# Main function
def main():
    input_train_folder = "n_train"
    input_valid_folder = "n_valid"
    output_train_folder = "Train"
    output_valid_folder = "Valid"

    # Apply histogram equalization to subfolders in n_train and save to final_train
    apply_histogram_equalization(input_train_folder, output_train_folder)

    # Apply histogram equalization to subfolders in n_valid and save to final_valid
    apply_histogram_equalization(input_valid_folder, output_valid_folder)

if __name__ == "__main__":
    main()
