import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from collections import deque

# -------------------------------
# Page Config & Custom Styling
# -------------------------------
st.set_page_config(page_title="👁️ Cataract Detection", page_icon="👁️", layout="wide")

st.markdown("""
    <style>
        /* Background Gradient */
        .main {
            background: linear-gradient(120deg, #e0f7fa, #fce4ec);
            font-family: 'Poppins', sans-serif;
        }
        /* Custom Fonts */
        @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600&display=swap');
        
        h1, h2, h3, h4 {
            font-family: 'Poppins', sans-serif;
            font-weight: 600;
            color: #1e3a8a;
        }
        p, li, span, div {
            font-family: 'Poppins', sans-serif;
        }
        /* Buttons */
        .stButton>button {
            background: linear-gradient(45deg, #2563eb, #1d4ed8);
            color: white;
            font-weight: bold;
            border-radius: 12px;
            padding: 0.6em 1.2em;
            font-size: 16px;
            border: none;
        }
        .stButton>button:hover {
            background: linear-gradient(45deg, #1e40af, #1d4ed8);
        }
        /* Prediction Card */
        .pred-card {
            padding: 20px;
            border-radius: 15px;
            text-align: center;
            font-size: 22px;
            font-weight: bold;
            margin-top: 20px;
        }
        .normal {
            background: #dcfce7;
            color: #166534;
            border: 2px solid #16a34a;
        }
        .cataract {
            background: #fee2e2;
            color: #b91c1c;
            border: 2px solid #dc2626;
        }
        /* Data Structures Section */
        .ds-section {
            background: #ffffff;
            padding: 15px;
            border-radius: 12px;
            box-shadow: 0 3px 8px rgba(0,0,0,0.1);
            margin-top: 25px;
        }
    </style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Instructions & About
# -------------------------------
st.sidebar.header("📌 Instructions")
st.sidebar.markdown("""
1. Upload an **eye image** (jpg/jpeg/png).  
2. Wait for the AI model to analyze.  
3. See the uploaded image + prediction result.  
4. Check **Hash Table** & **Queue** to view how DS concepts are applied.  
5. Use **Clear Queue & Hash Table** to reset.
""")

st.sidebar.header("ℹ️ About Project")
st.sidebar.markdown("""
**Project Title:** Cataract Detection using CNN & DS Concepts  
**Developed by:** Nandini  
**Concepts Used:**  
- Convolutional Neural Networks (CNN)  
- Data Structures (Hash Table + Queue)  
- Streamlit for Web App Deployment  
""")

# Optional Report Download
report_path = "report.pdf"  # place your report in this folder
if os.path.exists(report_path):
    with open(report_path, "rb") as f:
        st.sidebar.download_button("📥 Download Report", f, file_name="Cataract_Project_Report.pdf")
else:
    st.sidebar.info("📑 Project report not uploaded yet.")

# -------------------------------
# Load Model
# -------------------------------
MODEL_PATH = "cataract_model.h5"
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH)
else:
    st.error("⚠️ Model file not found! Please upload `cataract_model.h5`.")
    st.stop()

# -------------------------------
# DS Structures
# -------------------------------
if "hash_table" not in st.session_state:
    st.session_state.hash_table = {}

if "queue" not in st.session_state:
    st.session_state.queue = deque(maxlen=10)

# -------------------------------
# Title & Description
# -------------------------------
st.title("👁️ Cataract Detection System")
st.markdown("### Upload an eye image to check whether it is **Normal** or has **Cataract**.")

# -------------------------------
# Upload & Prediction
# -------------------------------
uploaded_file = st.file_uploader("📂 Upload Eye Image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess
    img = image.load_img(uploaded_file, target_size=(150,150))
    img_array = image.img_to_array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]
    label = "Cataract" if pred > 0.5 else "Normal"

    # Show Styled Prediction
    if label == "Normal":
        st.markdown(f"<div class='pred-card normal'>✅ Eye is Normal</div>", unsafe_allow_html=True)
    else:
        st.markdown(f"<div class='pred-card cataract'>⚠️ Cataract Detected!</div>", unsafe_allow_html=True)

    # Update DS Structures
    st.session_state.hash_table[uploaded_file.name] = label
    st.session_state.queue.append((uploaded_file.name, label))

# -------------------------------
# DS Visualization
# -------------------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown("<div class='ds-section'><h4>🔑 Hash Table</h4>", unsafe_allow_html=True)
    if st.session_state.hash_table:
        st.json(st.session_state.hash_table)
    else:
        st.info("No predictions yet!")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown("<div class='ds-section'><h4>📋 FIFO Queue (Recent Predictions)</h4>", unsafe_allow_html=True)
    if st.session_state.queue:
        st.write(list(st.session_state.queue))
    else:
        st.info("Queue is empty!")
    st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------
# Reset Button
# -------------------------------
if st.button("🗑️ Clear Queue & Hash Table"):
    st.session_state.hash_table.clear()
    st.session_state.queue.clear()
    st.success("✅ Cleared all stored predictions!")

# -------------------------------
# Footer
# -------------------------------
st.markdown("---")
st.markdown("<center>👩‍💻 Developed by: <b>Nandini</b> <br>📌 Project: Cataract Detection with CNN, Streamlit & DS</center>", unsafe_allow_html=True)
