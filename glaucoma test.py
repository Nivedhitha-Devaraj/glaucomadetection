from PIL import Image
import numpy as np
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('glaucoma_detection_model.keras')

# Path to the image you want to test
img_path =r'C:\Users\kukum\OneDrive\Documents\testing\glaucoma\Im312_g_ACRIMA.jpg'  # Ensure this is correct

# Function to preprocess the image (resizing, normalization, and expanding dimensions)
def preprocess_image(img_path, target_size=(224, 224)):
    try:
        img = Image.open(img_path)  # Open the image file
        img = img.resize(target_size)  # Resize image to the target size (224x224 in this case)
        img_array = np.array(img)  # Convert image to array format
        img_array = img_array / 255.0  # Normalize pixel values (scale from 0-255 to 0-1)
        img_array = np.expand_dims(img_array, axis=0)  # Add a batch dimension for the model (1, 224, 224, 3)
        return img_array
    except Exception as e:
        print(f"Error processing image: {e}")
        return None

# Preprocess the image
img_array = preprocess_image(img_path)

if img_array is not None:
    # Make predictions using the model
    predictions = model.predict(img_array)
    predicted_prob = predictions[0][0]  # Get the predicted probability

    # Output the predicted probability
    print(f"Predicted Probability: {predicted_prob:.4f}")

    # Classify the image based on the predicted probability (you can adjust the threshold)
    threshold = 0.5  # You can change this threshold if needed
    predicted_class = (predicted_prob > threshold).astype("int32")

    # Output the result based on the threshold
    if predicted_class == 1:
        print(f"The image is classified as Normal with a probability of {predicted_prob:.4f}")
    else:
        print(f"The image is classified as Glaucoma with a probability of {predicted_prob:.4f}")

    # Display model accuracy as a reference
    test_accuracy = 0.90 # Update with your actual accuracy from training
    print(f"Model Accuracy: {test_accuracy * 100:.2f}%")
else:
    print("Image could not be processed. Please check the image path or format.")
