# ❤️ Heart_Disease_Prediction_App

An end-to-end machine learning system that predicts the likelihood of heart disease using patient clinical data. This project features a live **Streamlit Web Application** that provides real-time predictions and visualizes individualized risk factors.

## 🚀 Live Demo
🔗 **https://heartdiseasepredictionapp-yzmg8ae3jznmmqusskdkc8.streamlit.app/**

---

## ✨ Features
- **Individual Patient Diagnostics:** Toggle medical parameters (Age, Cholesterol, Resting BP) to receive an instant risk percentage.

<img width="748" height="636" alt="image" src="https://github.com/user-attachments/assets/df9aa8a0-adcf-4dc9-b24a-138bcea53ea2" />

<img width="883" height="640" alt="image" src="https://github.com/user-attachments/assets/d8edb98b-d94d-459c-8375-fa65fc9d6ee0" />

- **Explainable AI (no zero-clutter):** Dynamic charts showing *exactly* which metrics pushed a patient's risk score up or down.

<img width="341" height="193" alt="image" src="https://github.com/user-attachments/assets/84043555-6bc3-4056-bc41-3835c593c85b" />

<img width="759" height="561" alt="image" src="https://github.com/user-attachments/assets/2ca70e02-2e95-4908-a692-ab3bf0c32c25" />

- **Pre-processed Pipeline:** Seamlessly handles One-Hot Encoding and Standard Scaling behind the scenes.

Want to run this app on your own computer? Follow these steps:

1. Clone the repository

git clone [https://github.com/KehindeOladele/Heart_Disease_Prediction_App.git](https://github.com/KehindeOladele/Heart_Disease_Prediction_App.git)
cd Heart_Disease_Prediction_App

2. Install dependencies

It is recommended to use a virtual environment. Install required libraries using:

pip install -r requirements.txt

3. Launch the App

Run the Streamlit app from your terminal:

streamlit run heartdisease/heart_disease_app.py

## 📊 Machine Learning Workflow
Exploratory Data Analysis (EDA): Correlation heatmaps were generated to find features most correlated with heart health.

Preprocessing: Outliers were processed, categorical variables one-hot encoded (to bypass the dummy variable trap), and numerical values standard-scaled.

Model Selection: Tested Logistic Regression, SVM, and Random Forest Classifier. The final ensemble model was selected for its high Recall rating (minimizing false negatives in clinical detection).

## 🛠️ Technology Stack
Languages: Python 🐍

Frameworks: Streamlit

Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Joblib

## 📂 Repository Structure

The core machine learning and app files live inside the `heartdisease/` directory:

---
```text
├── heartdisease/
│   ├── heart_disease_app.py      # Streamlit application script
│   ├── heart_disease_model.pkl   # Trained Random Forest Model
│   ├── scaler.pkl                # Fitted StandardScaler
│   ├── feature_names.pkl         # Saved One-Hot encoded column names
│   └── Heart_Disease_Prediction_Model.ipynb # Jupyter Notebook (EDA & Training)
├── requirements.txt              # Required python libraries
└── README.md                     # Project Documentation
