# Import necessary libraries
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import base64

# Set page title and icon as the first Streamlit command
st.set_page_config(
    page_title="Ocular Disease Recognition",
    page_icon="👁️",
    layout="wide",  # Adjust layout for better spacing and layout
)

# Define your preprocessing function based on ResNet50 preprocessing
def preprocess_single_image(image):
    # Resize the image to match the input size of your model (224x224)
    image = image.resize((224, 224))
    image = tf.keras.applications.resnet50.preprocess_input(np.array(image))
    return image

# Load the saved model
model = tf.keras.models.load_model(r"C:\Users\anura\OneDrive\Documents\Projects\Ocular_disease.h5")  # Replace with your actual model path

# Define a function for styling elements with CSS
def set_css_style():
    st.markdown(
        f"""
        <style>
            .stApp {{
                background: linear-gradient(135deg, #8AB6D6, #C9E4ED); /* Gradient colors */
                font-family: 'Arial', sans-serif;
                color: black; /* Text color on the gradient background */
            }}
            .st-bd {{
                padding: 2rem;
            }}
            .stTitle {{
                font-size: 3rem;
                text-align: center;
                margin-bottom: 1rem;
                color: #007ACC; /* Title color */
            }}
            .stText {{
                font-size: 1.5rem;
                text-align: center;
                margin-bottom: 1.5rem;
                color: #333; /* Text color */
            }}
            .stImageWrapper {{
                margin: 2rem 0;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
            }}
            .stImage {{
                max-width: 100%; /* Adjust the maximum width as needed */
                border-radius: 10px;
            }}
            .stButton {{
                background-color: #007ACC;
                color: white;
                padding: 1rem 2rem;
                border-radius: 0.5rem;
                text-align: center;
                font-weight: bold;
                font-size: 1.2rem;
                transition: background-color 0.3s;
            }}
            .stButton:hover {{
                background-color: #005A8C;
            }}
            .stResult {{
                text-align: center;
                font-size: 2rem;
                font-weight: bold;
                margin-top: 2rem;
                color: #007ACC; /* Result color */
            }}
            .stFooter {{
                text-align: center;
                margin-top: 2rem;
                font-size: 1.5rem;
                color: #333; /* Footer text color */
            }}
            .stCreators {{
                display: flex;
                justify-content: center;
                margin: 2rem 0;
                color: #007ACC; /* Creator text color */
            }}
            .stCreator {{
                text-align: center;
                margin: 0 1rem;
            }}
            .stDisclaimer {{
                text-align: center;
                font-size: 1rem;
                color: #FF0000;
                margin-top: 1rem;
            }}
            .stDivider {{
                margin-top: 2rem;
                margin-bottom: 2rem;
                border: 1px solid #007ACC;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )

# Apply the custom CSS style
set_css_style()

# Continue with the rest of your Streamlit app
st.title('Ocular Disease Recognition')
st.write("Upload an eye image to classify if it's 'Disease' or 'No Disease.")

# Upload an image for prediction
uploaded_image = st.file_uploader("Choose an eye image...", type=["jpg", "jpeg", "png"])

# Add disclaimer about model predictions
st.markdown(
    """
    <div class="stDisclaimer">
        <p><b>Disclaimer:</b> This model's predictions are for informational purposes only. 
        Consult with a medical professional for accurate diagnosis and treatment.</p>
    </div>
    """,
    unsafe_allow_html=True
)

if uploaded_image is not None:
    # Display the uploaded image with visual spacing
    image_data = uploaded_image.read()
    encoded_image = base64.b64encode(image_data).decode()
    st.markdown(
        f'<div class="stImageWrapper"><img src="data:image/jpeg;base64,{encoded_image}" alt="Uploaded Image" class="stImage" /></div>',
        unsafe_allow_html=True
    )

    # Add spacing for visual separation
    st.write("")

    # Preprocess the image and make predictions
    image = Image.open(uploaded_image)
    image = preprocess_single_image(image)
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Make a prediction
    prediction = model.predict(image)

    # You'll get a probability value; you can threshold it for binary classification
    if prediction > 0.5:
        predicted_class = 'Disease'  # Your positive class
        confidence = prediction[0][0]
    else:
        predicted_class = 'No Disease'  # Your negative class
        confidence = 1 - prediction[0][0]

    # Display the prediction result with styling and visual feedback
    st.markdown('<hr class="stDivider">', unsafe_allow_html=True)  # Add a horizontal line as a divider
    st.write("### Prediction Result:")
    st.write(f"**Predicted Class:** {predicted_class}")
    st.write(f"**Confidence:** {confidence:.2%}", key="stResult")

    # Additional details about the prediction
    st.markdown('<div class="stDetails">', unsafe_allow_html=True)
    st.write("#### Additional Details:")
    st.write(f"- Probability of Disease: {prediction[0][0]:.4f}")
    st.write(f"- Probability of No Disease: {1 - prediction[0][0]:.4f}")
    # Add more details as needed
    st.markdown('</div>', unsafe_allow_html=True)

# Add user-friendly messages and error handling
if uploaded_image is None:
    st.info("Please upload an eye image to classify if it's 'Disease' or 'No Disease'.")
elif not uploaded_image.type.startswith("image/"):
    st.error("Please upload a valid image file (JPEG or PNG).")

# Add a footer with custom background, better typography, and information
st.markdown('<hr class="stDivider">', unsafe_allow_html=True)  # Add a horizontal line as a divider
st.markdown('<div class="stFooter"> Ocular Disease Recognition </div>', unsafe_allow_html=True)
st.markdown('<div class="stCreators">Created by Anurag Nandanwar</div>', unsafe_allow_html=True)
