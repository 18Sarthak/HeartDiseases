import streamlit as st
import numpy as np
import joblib

# Load model and scaler
model = joblib.load("heart_disease_model.pkl")
scaler = joblib.load("scaler.pkl")

st.set_page_config(page_title="Heart Disease Predictor", layout="centered")
st.title("💓 Heart Disease Prediction App")
st.markdown("Predict the risk of heart disease using patient health data.")

# Input form
with st.form("user_input_form"):
    age = st.slider("Age", 20, 80, 50)
    sex = st.radio("Sex", ["Male", "Female"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure (mmHg)", 90, 200, 120)
    chol = st.number_input("Serum Cholesterol (mg/dL)", 100, 600, 200)
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dL", ["Yes", "No"])
    restecg = st.selectbox("Resting ECG Results", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
    thalach = st.slider("Maximum Heart Rate Achieved", 70, 210, 150)
    exang = st.radio("Exercise Induced Angina", ["Yes", "No"])
    oldpeak = st.slider("ST Depression (oldpeak)", 0.0, 6.5, 1.0)
    slope = st.selectbox("Slope of ST Segment", ["Upsloping", "Flat", "Downsloping"])
    
    submit = st.form_submit_button("🔍 Predict")

if submit:
    # Binary encodings
    sex_val = 1 if sex == "Male" else 0
    fbs_val = 1 if fbs == "Yes" else 0
    exang_val = 1 if exang == "Yes" else 0

    # One-hot encoding for cp
    cp_1 = 1 if cp == "Atypical Angina" else 0
    cp_2 = 1 if cp == "Non-anginal Pain" else 0
    cp_3 = 1 if cp == "Asymptomatic" else 0

    # One-hot encoding for restecg
    restecg_1 = 1 if restecg == "ST-T Wave Abnormality" else 0
    restecg_2 = 1 if restecg == "Left Ventricular Hypertrophy" else 0

    # One-hot encoding for slope
    slope_1 = 1 if slope == "Flat" else 0
    slope_2 = 1 if slope == "Downsloping" else 0

    # Final input array (19 features)
    input_features = np.array([[age, sex_val, trestbps, chol, fbs_val,
                               restecg_1, restecg_2, thalach, exang_val, 
                               oldpeak, slope_1, slope_2, cp_1, cp_2, cp_3,
                               0, 0, 0, 0]])

    # Scale and predict
    scaled_input = scaler.transform(input_features)
    prediction = model.predict(scaled_input)[0]
    prediction_proba = model.predict_proba(scaled_input)[0]

    # Show probability percentages
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Probability of No Heart Disease", f"{prediction_proba[0]*100:.2f}%")
    with col2:
        st.metric("Probability of Heart Disease", f"{prediction_proba[1]*100:.2f}%")

    # Show final result
    if prediction == 1:
        st.error("⚠️ High Risk: Heart Disease Detected")
    else:
        st.success("✅ Low Risk: No Heart Disease")