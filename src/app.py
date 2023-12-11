import streamlit as st
import pickle
import pandas as pd
import numpy as np

# About Section with Style
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app predicts sepsis based on medical input data. "
    "It uses a machine learning model trained on a dataset of sepsis cases."
)

# Welcome Message with Style
st.write(
    "👋 Welcome to the Sepsis Prediction App! Enter the medical data in the sidebar, "
    "click 'Predict Sepsis', and get the prediction result."
)

# Load the model and key components
with open('model_and_key_components.pkl', 'rb') as file:
    loaded_components = pickle.load(file)

loaded_model = loaded_components['model']
loaded_encoder = loaded_components['encoder']
loaded_scaler = loaded_components['scaler']

# Data Fields
data_fields = {
    "PRG": "Number of pregnancies (applicable only to females)",
    "PL": "Plasma glucose concentration (mg/dL)",
    "PR": "Diastolic blood pressure (mm Hg)",
    "SK": "Triceps skinfold thickness (mm)",
    "TS": "2-hour serum insulin (mu U/ml)",
    "M11": "Body mass index (BMI) (weight in kg / {(height in m)}^2)",
    "BD2": "Diabetes pedigree function (mu U/ml)",
    "Age": "Age of the patient (years)"
}

# Page Title with Style
st.title("🩸 Sepsis Prediction App")
st.markdown("---")

# Sidebar with Data Fields
st.sidebar.title("📊 Input Data")
input_data = {}
for field, description in data_fields.items():
    input_data[field] = st.sidebar.number_input(description, value=0.0)

# Function to preprocess input data
def preprocess_input_data(input_data):
    numerical_cols = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']
    input_data_scaled = loaded_scaler.transform([list(input_data.values())])
    return pd.DataFrame(input_data_scaled, columns=numerical_cols)

# Function to make predictions
def make_predictions(input_data_scaled_df):
    y_pred = loaded_model.predict(input_data_scaled_df)
    sepsis_mapping = {0: 'Negative', 1: 'Positive'}
    return sepsis_mapping[y_pred[0]]

# Predict Button with Style
if st.sidebar.button("🔮 Predict Sepsis"):
    try:
        input_data_scaled_df = preprocess_input_data(input_data)
        sepsis_status = make_predictions(input_data_scaled_df)
        st.success(f"The predicted sepsis status is: {sepsis_status}")
    except Exception as e:
        st.error(f"An error occurred: {e}")

# Display Data Fields and Descriptions
st.sidebar.title("🔍 Data Fields")
for field, description in data_fields.items():
    st.sidebar.text(f"{field}: {description}")