import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from keras.preprocessing.image import img_to_array

# Load the trained model
model_path = "/content/drive/MyDrive/Plant_Disease_Dataset/plant_disease_model.keras"
model = tf.keras.models.load_model(model_path)

# Class labels
class_labels = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___healthy']

# Care Recommendations Dictionary
care_recommendations = {
    'Potato___Early_blight': [
        "Remove and destroy infected leaves.",
        "Avoid overhead watering.",
        "Use fungicides if necessary."
    ],
    'Potato___Late_blight': [
        "Remove infected plants immediately.",
        "Use resistant potato varieties.",
        "Apply appropriate fungicide."
    ],
    'Potato___healthy': [
        "Maintain regular watering.",
        "Ensure proper sunlight.",
        "Watch for early signs of disease."
    ],
    'Tomato___Bacterial_spot': [
        "Remove infected leaves and fruits.",
        "Avoid wetting leaves during watering.",
        "Apply copper-based bactericides."
    ],
    'Tomato___Early_blight': [
        "Remove lower infected leaves.",
        "Provide good air circulation.",
        "Use fungicides if needed."
    ],
    'Tomato___Late_blight': [
        "Destroy infected plants.",
        "Avoid wet foliage.",
        "Use resistant tomato varieties."
    ],
    'Tomato___healthy': [
        "Provide balanced nutrients.",
        "Maintain clean growing environment.",
        "Monitor regularly for pests."
    ]
}

# Preprocess image
def preprocess_image(image):
    image = cv2.resize(image, (256, 256))
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Set UI config
st.set_page_config(page_title="Plant Doctor", page_icon="🌿", layout="centered")

# Custom CSS intact
st.markdown("""
    <style>
        .header {
            text-align: center;
            padding: 1.5rem 0;
        }
        .header h1, .header h2 {
            color: #2e8b57;
            margin-bottom: 0.5rem;
        }
        .upload-area {
            border: 2px dashed #2e8b57;
            border-radius: 12px;
            padding: 2rem;
            text-align: center;
            margin: 1.5rem 0;
            background-color: #f8f9fa;
        }
        .result-card {
            background: white;
            border-radius: 12px;
            padding: 1.5rem;
            margin: 1.5rem 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .healthy {
            border-left: 5px solid #4CAF50;
        }
        .diseased {
            border-left: 5px solid #FF5722;
        }
        .confidence-meter {
            height: 8px;
            background: #e0e0e0;
            border-radius: 4px;
            margin: 0.5rem 0 1rem;
            overflow: hidden;
        }
        .confidence-fill {
            height: 100%;
            background: #2e8b57;
        }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="header">
        <h1>🌿 Botanical Care</h1>
        <h2> A Plant Disease Detection System</h2>
        <p>Upload a leaf image to detect potential diseases</p>
    </div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"], label_visibility="collapsed")
st.markdown('</div>', unsafe_allow_html=True)

if uploaded_file is not None:
    with st.spinner('Analyzing your plant...'):
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        st.image(image, caption='Uploaded Image', use_column_width=True)

        processed_image = preprocess_image(image)
        prediction = model.predict(processed_image)
        predicted_class = class_labels[np.argmax(prediction)]
        confidence = np.max(prediction) * 100

        # Confidence Threshold
        threshold = 60  # percent

        st.markdown(f'<div class="result-card {"healthy" if "healthy" in predicted_class else "diseased"}">', unsafe_allow_html=True)

        # If confidence low
        if confidence < threshold:
            st.warning("Unable to confidently detect the disease. Please upload a clearer image of Potato or Tomato leaf.")
        
        # If unknown plant (not Potato or Tomato)
        elif not (predicted_class.startswith("Potato") or predicted_class.startswith("Tomato")):
            st.warning("This system is designed to detect diseases in Potato and Tomato leaves only. Please upload a valid image.")

        # Valid Prediction
        else:
            if "healthy" in predicted_class:
                st.success("**Healthy Plant Detected**")
            else:
                st.error(f"**Disease Detected:** {predicted_class.replace('___', ' ').title()}")

            st.write(f"**Confidence:** {confidence:.1f}%")
            st.markdown(f"""
                <div class="confidence-meter">
                    <div class="confidence-fill" style="width:{confidence}%"></div>
                </div>
            """, unsafe_allow_html=True)

            with st.expander("📌 Care Recommendations"):
                for tip in care_recommendations[predicted_class]:
                    st.write(f"- {tip}")

        st.markdown('</div>', unsafe_allow_html=True)

st.markdown("""
    <div style="text-align: center; color: #666; margin-top: 2rem;">
        <small>For accurate diagnosis, consult with a plant specialist</small>
    </div>
""", unsafe_allow_html=True)