import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os

# Load trained model
model = load_model("cataract_model.h5")

# Folder containing test images
current_dir = os.path.dirname(os.path.abspath(__file__))
test_folder = os.path.join(current_dir, "test_images")  # put all test images in this folder

# Create folder if it doesn't exist
if not os.path.exists(test_folder):
    os.makedirs(test_folder)
    print(f"📁 Please add test images into this folder: {test_folder}")
    exit()

# Get all image files in the folder
image_files = [f for f in os.listdir(test_folder) if f.lower().endswith((".jpg", ".jpeg", ".png"))]

if not image_files:
    print("⚠️ No images found in the test folder!")
    exit()

def predict_eye(img_path):
    # Load and resize image
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Make prediction
    prediction = model.predict(img_array)

    # Convert prediction to label
    label = "Cataract" if prediction[0][0] > 0.5 else "Normal"
    print(f"{os.path.basename(img_path)}: {label}")

# Predict all images
for img_file in image_files:
    img_path = os.path.join(test_folder, img_file)
    predict_eye(img_path)




