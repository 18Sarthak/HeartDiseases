import streamlit as st
import numpy as np
import sys
import subprocess

# Package installation verification
try:
    import joblib
except ImportError:
    st.warning("Installing required packages...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib==1.3.2"])
    import joblib

# Load model files with error handling
@st.cache_resource
def load_model():
    try:
        model = joblib.load("heart_disease_model.pkl")
        scaler = joblib.load("scaler.pkl")
        return model, scaler
    except Exception as e:
        st.error(f"""
        ❌ Error loading model files: {str(e)}
        
        Please ensure:
        1. 'heart_disease_model.pkl' exists
        2. 'scaler.pkl' exists
        3. Files are in the same directory as app.py
        """)
        st.stop()

model, scaler = load_model()

# App UI
st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("Predict the risk of heart disease using patient health data.")

with st.form("user_input_form"):
    st.subheader("Patient Information")
    age = st.slider("Age", 20, 80, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    
    st.subheader("Medical Measurements")
    col1, col2 = st.columns(2)
    with col1:
        trestbps = st.number_input("Resting BP (mmHg)", 90, 200, 120)
    with col2:
        chol = st.number_input("Cholesterol (mg/dL)", 100, 600, 200)
    
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.slider("Max Heart Rate", 70, 210, 150)
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression", 0.0, 6.5, 1.0, step=0.1)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    submit = st.form_submit_button("🔍 Predict Risk")

if submit:
    # Feature encoding
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0
    
    # One-hot encodings
    cp_1 = 1 if cp == "Atypical Angina" else 0
    cp_2 = 1 if cp == "Non-anginal Pain" else 0
    cp_3 = 1 if cp == "Asymptomatic" else 0
    
    restecg_1 = 1 if restecg == "ST-T Wave Abnormality" else 0
    restecg_2 = 1 if restecg == "Left Ventricular Hypertrophy" else 0
    
    slope_1 = 1 if slope == "Flat" else 0
    slope_2 = 1 if slope == "Downsloping" else 0

    # Create feature array (19 features)
    input_features = np.array([[age, sex_val, trestbps, chol, fbs_val,
                              restecg_1, restecg_2, thalach, exang_val,
                              oldpeak, slope_1, slope_2, cp_1, cp_2, cp_3,
                              0, 0, 0, 0]])  # Placeholders for any missing features

    try:
        # Scale and predict
        scaled_input = scaler.transform(input_features)
        prediction = model.predict(scaled_input)[0]
        prediction_proba = model.predict_proba(scaled_input)[0]

        # Display results
        st.divider()
        st.subheader("Prediction Results")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Healthy Probability", f"{prediction_proba[0]*100:.1f}%", 
                      help="Probability of no heart disease")
        with col2:
            st.metric("Risk Probability", f"{prediction_proba[1]*100:.1f}%",
                     help="Probability of heart disease presence")

        # Final verdict
        if prediction == 1:
            st.error("⚠️ High Risk: Heart Disease Detected", icon="🚨")
            st.warning("Consult a cardiologist for further evaluation")
        else:
            st.success("✅ Low Risk: No Heart Disease Detected", icon="✅")
            st.info("Maintain regular heart health checkups")

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.info("Please check your input values and try again")

# Add footer
st.divider()
st.caption("Note: This prediction is for informational purposes only and should not replace professional medical advice.")