import streamlit as st
import numpy as np
import joblib

# ===============================
# PAGE CONFIG
# ===============================
st.set_page_config(
    page_title="Titanic Survival Predictor",
    page_icon="🚢",
    layout="wide"
)

# ===============================
# CUSTOM CSS
# ===============================
st.markdown("""
<style>
.stApp {
    background: linear-gradient(to right, #000428, #004e92);
    color: white;
}

h1 {
    text-align: center;
    color: #FFD700;
}

.prediction-box {
    padding: 20px;
    background-color: white;
    color: black;
    border-radius: 12px;
    text-align: center;
    font-size: 24px;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

st.title("🚢 Titanic Survival Prediction App")

# ===============================
# LOAD TRAINED MODEL
# ===============================
model = joblib.load("model_kbest.pkl")

# ===============================
# INPUT FEATURES
# (Replace with your actual selected 3 features)
# ===============================

st.subheader("🧍 Enter Passenger Details")

# Example selected features (change if needed)
pclass = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
sex = st.selectbox("Sex", ["Male", "Female"])
fare = st.number_input("Fare", min_value=0.0)

# Encode Sex same as training
sex_encoded = 1 if sex == "male" or sex == "Male" else 2

# Create input array (must match training order)
input_data = np.array([[pclass, sex_encoded, fare]])

# ===============================
# PREDICTION
# ===============================
if st.button("Predict Survival"):

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        result = "🎉 Survived"
    else:
        result = "💀 Did Not Survive"

    st.markdown(f"""
    <div class="prediction-box">
        Prediction Result: {result}
    </div>
    """, unsafe_allow_html=True)