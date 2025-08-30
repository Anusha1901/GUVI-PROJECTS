import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load trained model
model = tf.keras.models.load_model("GARBAGE_classifier.h5")

# Class names (must match training order)
class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]

# Sidebar
st.sidebar.title("♻️ RecycleVision")
st.sidebar.info(
    """
    📌 Upload a garbage image  
    📌 Our AI model will classify it  
    📌 Categories: Cardboard, Glass, Metal, Paper, Plastic, Trash
    """
)

# Main title
st.title("♻️ Garbage Classification App")
st.markdown("### Upload an image and let AI predict the type of waste 🗑️")

# File uploader
uploaded_file = st.file_uploader("📤 Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="🖼️ Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.resize((224, 224))   # Resize to match model input
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    predictions = model.predict(img_array)
    predicted_class = class_names[np.argmax(predictions)]
    confidence = np.max(predictions) * 100

    # Show result
    st.success(f"✅ Prediction: **{predicted_class}**")
    st.info(f"📊 Confidence: **{confidence:.2f}%**")

    # Bar chart of probabilities
    fig, ax = plt.subplots()
    ax.bar(class_names, predictions[0], color="skyblue")
    ax.set_ylabel("Probability")
    ax.set_xlabel("Class")
    ax.set_title("Prediction Probabilities")
    plt.xticks(rotation=30)

    st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("🚀 Built with ❤️ using **Streamlit + TensorFlow**")