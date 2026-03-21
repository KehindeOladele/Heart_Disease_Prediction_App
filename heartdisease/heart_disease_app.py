import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Streamlit app
st.title("Heart Disease Prediction App")
st.write("Enter the following details to predict the likelihood of heart disease:")

# Create input fields
age = st.number_input("Age", min_value=1, max_value=120, value=30)
sex = st.selectbox("Sex", options=["Male", "Female"])
chest_pain_type = st.selectbox("Chest Pain Type", options=["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"])
resting_blood_pressure = st.number_input("Resting Blood Pressure", min_value=1, max_value=200, value=120)
cholesterol = st.number_input("Cholesterol", min_value=1, max_value=400, value=200)
fasting_blood_sugar = st.selectbox("Fasting Blood Sugar", options=["> 120 mg/dl", "< 120 mg/dl"])
rest_ecg = st.selectbox("Resting ECG", options=["Normal", "ST-T Wave Abnormality", "Left Ventricular Hypertrophy"])
Max_heart_rate = st.number_input("Max Heart Rate", min_value=1, max_value=250, value=150)
exercise_induced_angina = st.selectbox("Exercise Induced Angina", options=["Yes", "No"])
st_slope = st.selectbox("ST Slope", options=["Upsloping", "Flat", "Downsloping"])
vessels_colored_by_flourosopy = st.number_input("vessels_colored_by_flourosopy", min_value=0, max_value=3, value=0)
thalassemia = st.selectbox("Thalassemia", options=["Normal", "Fixed Defect", "Reversible Defect"])
oldpeak = st.number_input("ST Depression Induced by Exercise", min_value=0.0, max_value=6.0, value=1.0)

# Use script directory so model loads work even when current working directory is different
ROOT = Path(__file__).resolve().parent

# Load the trained model, scaler and feature names
model = joblib.load(ROOT / 'heart_disease_model.pkl')
scaler = joblib.load(ROOT / 'scaler.pkl')
feature_names = joblib.load(ROOT / 'feature_names.pkl')

# input handler
def process_input(input_df, feature_names):
    # 1. Convert categorical to strings (matching your notebook's astype(str))
    cat_cols = ['sex', 'chest_pain_type', 'fasting_blood_sugar', 
                'rest_ecg', 'exercise_induced_angina', 'st_slope', 
                'vessels_colored_by_flourosopy', 'thalassemia']

    input_encoded = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)
    # 2. Add missing columns with default value 0
    for col in feature_names:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    # 3. Reorder columns to match training data
    input_encoded = input_encoded[feature_names]
    # 4. Scale numerical features
    num_cols = ['age', 'resting_blood_pressure', 'cholesterol', 'Max_heart_rate', 'oldpeak']
    input_encoded[num_cols] = scaler.transform(input_encoded[num_cols])
    return input_encoded

# Create a DataFrame from user input
input_data = {
    "age": [age],
    "sex": [sex],
    "chest_pain_type": [chest_pain_type],
    "resting_blood_pressure": [resting_blood_pressure],
    "cholesterol": [cholesterol],
    "fasting_blood_sugar": [fasting_blood_sugar],
    "rest_ecg": [rest_ecg],
    "Max_heart_rate": [Max_heart_rate],
    "exercise_induced_angina": [exercise_induced_angina],
    "st_slope": [st_slope],
    "vessels_colored_by_flourosopy": [vessels_colored_by_flourosopy],
    "thalassemia": [thalassemia],
    "oldpeak": [oldpeak]
}
input_df = pd.DataFrame(input_data)

# Process input and make prediction
if st.button("Predict"):
    processed_input = process_input(input_df, feature_names)
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]  # Probability of heart disease
    st.write(f"Prediction: {'Heart Disease' if prediction == 1 else 'No Heart Disease'}")
    st.write(f"Probability of Heart Disease: {probability:.2f}")

# Show feature importance analysis for each person only contributing features that are relevant to the person
if st.button("Person-Specific Feature Analysis"):
    processed_input = process_input(input_df, feature_names)
    importances = model.feature_importances_
    contribution = processed_input.iloc[0] * importances
    # Filter out features that do not contribute
    non_zero_mask = contribution != 0
    contribution = contribution[non_zero_mask]
    filtered_feature_names = [name for i, name in enumerate(feature_names) if non_zero_mask[i]]
    feature_contribution = pd.DataFrame({
        'Feature': filtered_feature_names,
        'Contribution': contribution
    }).sort_values(by='Contribution', ascending=False)
    st.write("Feature Contribution to Prediction:")
    st.dataframe(feature_contribution)
    plt.figure(figsize=(8, 6))
    #must be non-negative for pie chart, so we take absolute value of contribution
    plt.pie(feature_contribution['Contribution'].abs(), labels=feature_contribution['Feature'], autopct='%1.1f%%')
    plt.title("Feature Contribution to Heart Disease Prediction")
    st.pyplot(plt)

