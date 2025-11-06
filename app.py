import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load trained model
model = joblib.load('model1.pkl')

st.title("❤️ Heart Disease Prediction App")

st.write("Enter the patient details below:")

# Input fields for each feature
age = st.number_input('Age', min_value=1, max_value=120, value=40)
sex = st.selectbox('Sex (1=Male, 0=Female)', [0, 1])
cp = st.selectbox('Chest Pain Type (0-3)', [0, 1, 2, 3])
trestbps = st.number_input('Resting Blood Pressure', value=120)
chol = st.number_input('Serum Cholestoral (mg/dl)', value=200)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (1=True, 0=False)', [0, 1])
restecg = st.selectbox('Resting ECG Results (0-2)', [0, 1, 2])
thalach = st.number_input('Max Heart Rate Achieved', value=150)
exang = st.selectbox('Exercise Induced Angina (1=Yes, 0=No)', [0, 1])
oldpeak = st.number_input('ST Depression Induced by Exercise', value=1.0)
slope = st.selectbox('Slope of Peak Exercise ST Segment (0-2)', [0, 1, 2])
ca = st.selectbox('Number of Major Vessels (0-3)', [0, 1, 2, 3])
thal = st.selectbox('Thalassemia (3=Normal, 6=Fixed Defect, 7=Reversible Defect)', [3, 6, 7])

# When button is clicked
if st.button("Predict"):
    input_data = np.array([[age, sex, cp, trestbps, chol, fbs, restecg,
                            thalach, exang, oldpeak, slope, ca, thal]])
    prediction = model.predict(input_data)

    if prediction[0] == 0:
        st.success("✅ No Heart Disease Detected")
    else:
        st.error("⚠️ Likely Heart Disease Detected")
