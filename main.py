import streamlit as st
import os
from PIL import Image
import tensorflow as tf
import numpy as np
from io import BytesIO

# Load your Keras model
MODEL = tf.keras.models.load_model('models/model.keras')

def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

load_css("style.css")

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

# TensorFlow logo path
tensorflow_logo = "logo.jpg"  # Path to your TensorFlow logo image

# Layout with two columns
col1, col2 = st.columns([9, 1])

with col1:
    st.header("TOMATO_DISEASE_DETECTION🍅")

with col2:
    st.image(tensorflow_logo, width=100, use_column_width=False)

# Add instructions
st.markdown("""
## Instructions🪴
1. Upload an image of a tomato leaf in JPG, JPEG, or PNG format.
2. Wait for the model to make a prediction.
3. View the predicted disease and its confidence score.
""")

# Sidebar for additional options
st.sidebar.header("Options")
show_more_probabilities = st.sidebar.checkbox("Show More Probabilities", value=False)

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
    
    # Display prediction probabilities if the option is selected
    if show_more_probabilities:
        st.sidebar.subheader("Prediction Probabilities")
        probabilities = {Class_Names[i]: predictions[0][i] for i in range(len(Class_Names))}
        st.sidebar.bar_chart(probabilities)

# Footer with model information and contact form
st.markdown("""
## Model Information
- **Model Architecture**: Used Tensorflow neural and convolution layers for prediction.
- **Training Data**: Describe the training data used for the model.
- **Performance Metrics**: .

## Contact
If you have any questions or feedback, please mail at: harshdipsaha95@gmail.com.
""")
