import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load trained model
model = load_model("cataract_model.h5")

# Page config
st.set_page_config(
    page_title="Cataract Detection",
    page_icon="👁️",
    layout="centered"
)

# Title and description
st.title("👁️ Cataract Detection App")
st.markdown("""
This app predicts whether an eye image has **Cataract** or is **Normal**.  
Upload a clear image of the eye, and the AI model will provide the prediction instantly.
""")

# Instructions in a sidebar
st.sidebar.header("Instructions")
st.sidebar.markdown("""
1. Click 'Browse files' and select an eye image (jpg/jpeg/png).  
2. Wait for the prediction.  
3. See the uploaded image and the prediction below.  
4. Recommended image size: **150x150 pixels or larger**.
""")

# Upload image
uploaded_file = st.file_uploader("Upload Eye Image", type=["jpg","jpeg","png"])

if uploaded_file:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Prepare image for prediction
    img = image.load_img(uploaded_file, target_size=(150,150))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Prediction
    prediction = model.predict(img_array)
    label = "Cataract" if prediction[0][0] > 0.5 else "Normal"

    # Show prediction
    st.markdown("### Prediction Result:")
    if label == "Cataract":
        st.success("⚠️ Cataract detected!")
    else:
        st.success("✅ Eye is Normal")

# Optional: Show training accuracy/loss graph if saved
if st.checkbox("Show Training Accuracy Graph"):
    st.subheader("Model Training Accuracy & Loss")
    # Example placeholder graph
    acc = [0.7, 0.85, 0.9, 0.93, 0.95]  # replace with your training history
    val_acc = [0.68, 0.83, 0.88, 0.91, 0.94]
    loss = [0.6,0.4,0.3,0.2,0.15]
    val_loss = [0.62,0.42,0.33,0.22,0.18]
    epochs = [1,2,3,4,5]

    fig, ax = plt.subplots(1,2, figsize=(12,4))
    ax[0].plot(epochs, acc, label='Train Accuracy')
    ax[0].plot(epochs, val_acc, label='Validation Accuracy')
    ax[0].set_title("Accuracy")
    ax[0].set_xlabel("Epoch")
    ax[0].set_ylabel("Accuracy")
    ax[0].legend()

    ax[1].plot(epochs, loss, label='Train Loss')
    ax[1].plot(epochs, val_loss, label='Validation Loss')
    ax[1].set_title("Loss")
    ax[1].set_xlabel("Epoch")
    ax[1].set_ylabel("Loss")
    ax[1].legend()

    st.pyplot(fig)

st.markdown("---")
st.markdown("Developed by: **Your Name**  \nProject: Cataract Detection using CNN & Streamlit")

