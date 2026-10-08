import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Heart Health AI", page_icon="❤️", layout="wide")

HEART_MODEL_PATH = "models/heart_risk_model.pkl"

@st.cache_resource
def load_heart_model():
    return joblib.load(HEART_MODEL_PATH)

st.title("❤️ Heart Health AI")
st.write("A machine-learning application that estimates cardiovascular risk from structured patient health information.")
st.info("This application is for educational purposes only and is not a medical diagnosis.")
st.divider()

st.header("Patient Information")
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=50)
    sex = st.selectbox("Sex", ["Female", "Male"])
    cp = st.selectbox("Chest Pain Type", ["Typical Angina", "Atypical Angina", "Non-anginal Pain", "Asymptomatic"])
    trestbps = st.number_input("Resting Blood Pressure", min_value=50, max_value=250, value=120)
    chol = st.number_input("Cholesterol", min_value=50, max_value=700, value=200)
    fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl", ["No", "Yes"])
    restecg = st.selectbox("Resting ECG", ["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])

with col2:
    thalach = st.number_input("Maximum Heart Rate", min_value=50, max_value=250, value=150)
    exang = st.selectbox("Exercise Induced Angina", ["No", "Yes"])
    oldpeak = st.number_input("ST Depression (Oldpeak)", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
    slope = st.selectbox("Slope", ["Upsloping", "Flat", "Downsloping"])
    ca = st.selectbox("Number of Major Vessels", [0, 1, 2, 3])
    thal = st.selectbox("Thalassemia", ["Normal", "Fixed Defect", "Reversible Defect"])

if st.button("Calculate Heart Risk", type="primary"):
    input_data = pd.DataFrame({
        "age": [age],
        "sex": [1 if sex == "Male" else 0],
        "cp": [{"Typical Angina": 0, "Atypical Angina": 1, "Non-anginal Pain": 2, "Asymptomatic": 3}[cp]],
        "trestbps": [trestbps],
        "chol": [chol],
        "fbs": [1 if fbs == "Yes" else 0],
        "restecg": [{"Normal": 0, "ST-T Wave Abnormality": 1, "Left Ventricular Hypertrophy": 2}[restecg]],
        "thalach": [thalach],
        "exang": [1 if exang == "Yes" else 0],
        "oldpeak": [oldpeak],
        "slope": [{"Upsloping": 0, "Flat": 1, "Downsloping": 2}[slope]],
        "ca": [ca],
        "thal": [{"Normal": 1, "Fixed Defect": 2, "Reversible Defect": 3}[thal]]
    })

    try:
        model = load_heart_model()
        prediction = model.predict(input_data)[0]

        probability = None
        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(input_data)[0][1]

        st.divider()
        st.header("Prediction Result")

        if probability is not None:
            st.metric("Estimated Cardiovascular Risk", f"{probability * 100:.2f}%")

        if prediction == 1:
            st.warning("The model estimates a higher cardiovascular-risk class.")
        else:
            st.success("The model estimates a lower cardiovascular-risk class.")

        st.caption("This is an ML estimate based on the trained CSV dataset model, not a medical diagnosis.")

    except Exception as e:
        st.error(f"Unable to make the prediction: {e}")