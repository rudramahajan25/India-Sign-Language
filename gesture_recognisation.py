import os
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model

# Load pre-trained model (You need to replace 'your_model.h5' with the actual path to your model)
model = load_model('D:\VIT\ISL detction project\model')

# Replace 'your_dataset_folder' with the actual path to your dataset folder
dataset_folder = 'dataset'

# Dictionary mapping class indices to letters
class_indices_to_letter = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J', 10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T', 20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'}

# Function to predict the letter from an image
def predict_letter(image_path):
    img = image.load_img(image_path, target_size=(64, 64))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    letter = class_indices_to_letter[predicted_class]
    return letter

# Loop through each folder (assuming each folder corresponds to a different letter)
for folder in os.listdir(dataset_folder):
    folder_path = os.path.join(dataset_folder, folder)
    if os.path.isdir(folder_path):
        print(f"Predictions for letter {folder}:")
        for file in os.listdir(folder_path):
            if file.endswith(('.jpg', '.jpeg', '.png')):
                file_path = os.path.join(folder_path, file)
                predicted_letter = predict_letter(file_path)
                print(f"File: {file}, Predicted Letter: {predicted_letter}")
        print("\n")
