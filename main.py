import streamlit as st
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO

# Load your Keras model
MODEL = tf.keras.models.load_model('models/model.keras')

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

# Define class names
Class_Names = [
    'Tomato_Bacterial_spot',
    'Tomato_Early_blight',
    'Tomato_Late_blight',
    'Tomato_Leaf_Mold',
    'Tomato_healthy'
]

# Set up directory to save images
if not os.path.exists("uploaded_images"):
    os.makedirs("uploaded_images")

# Title of the app
st.title("TOMATO_DISEASE_DETECTION")

# File uploader for image with extended text
uploaded_file = st.file_uploader("Choose an image in JPG/JPEG/PNG format...", type=["jpg", "jpeg", "png"])

# Display the uploaded image and save it
if uploaded_file is not None:
    # Read the uploaded file once
    file_data = uploaded_file.read()
    
    # Load the image using PIL
    image = Image.open(BytesIO(file_data))
    
    # Display the image
    st.image(image, caption="Uploaded Image", use_column_width=True)
    
    # Save the image
    image.save(os.path.join("uploaded_images", uploaded_file.name))
    
    # Process the image for prediction
    image_array = read_file_as_image(file_data)
    img_batch = np.expand_dims(image_array, 0)
    
    # Make predictions
    predictions = MODEL.predict(img_batch)
    predicted_class = Class_Names[np.argmax(predictions[0])]
    confidence = np.max(predictions[0])
    
    # Display the prediction results
    if confidence < 0.5:
        st.write("New disease detected: research needed")
    else:
        st.write(f"Prediction: {predicted_class} with confidence {confidence:.2f}")
