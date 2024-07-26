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
col1, col2 = st.columns([12, 1])

with col1:
    st.markdown('<h1 class="tomato-disease-header">TOMATO_DISEASE_DETECTION🍅</h1>', unsafe_allow_html=True)

with col2:
    st.markdown(f'<a href="https://www.tensorflow.org/api_docs/python/tf/all_symbols"><img src="{tensorflow_logo}" width="100"></a>', unsafe_allow_html=True)

# Add instructions
st.markdown('<h2 class="instructions-header">Instructions🪴</h2>', unsafe_allow_html=True)
st.markdown("""
1. Upload an image of a tomato leaf in JPG, JPEG, or PNG format.
2. Wait for the model to make a prediction.
3. View the predicted disease and its confidence score.
""")

# Sidebar for additional options
st.sidebar.markdown('<h2 class="options-header">Options</h2>', unsafe_allow_html=True)
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
    
    # Display the prediction results with animation
    st.components.v1.html(f"""
    <div style="text-align: center; margin-top: 20px;">
        <h2 style="font-size: 2.5em; font-weight: bold; color: goldenrod; animation: fadeIn 2s;">Prediction:</h2>
        <h3 style="font-size: 2em; font-weight: bold; color: darkgreen; animation: fadeIn 2s;">{predicted_class}</h3>
        <h3 style="font-size: 2em; font-weight: bold; color: violet; animation: fadeIn 3s;">Confidence:</h3>
        <h3 style="font-size: 2em; font-weight: bold; color: gray; animation: fadeIn 3s;">{confidence:.2f}</h3>
    </div>
    <style>
    @keyframes fadeIn {{
        0% {{ opacity: 0; }}
        100% {{ opacity: 1; }}
    }}
    </style>
    """, height=300)
    
    # Display prediction probabilities if the option is selected
    if show_more_probabilities:
        st.sidebar.subheader("Prediction Probabilities")
        probabilities = {Class_Names[i]: predictions[0][i] for i in range(len(Class_Names))}
        st.sidebar.bar_chart(probabilities)

# Footer with model information and contact form
st.markdown('<h2 class="model-info-header">Model Information</h2>', unsafe_allow_html=True)
st.markdown("""
- **Model Architecture**: Used Tensorflow neural and convolution layers for prediction.
- **Training Data**: [Kaggle Dataset](https://www.kaggle.com/datasets/arjuntejaswi/plant-village).
- **Performance Metrics**: [Sparse Categorical Cross entropy](https://fmorenovr.medium.com/sparse-categorical-cross-entropy-vs-categorical-cross-entropy-ea01d0392d28).
""")

st.markdown('<h2 class="contact-header">Contact</h2>', unsafe_allow_html=True)
st.markdown("""
If you have any questions or feedback, please mail at: harshdipsaha95@gmail.com.
""")
