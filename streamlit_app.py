import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# --- Load Model Artifacts ---
#@st.cache_resource
def load_model_artifacts():
    base_path = "model_artifacts"
    
    model = joblib.load(os.path.join(base_path, "random_forest_model.joblib"))
    
    with open(os.path.join(base_path, "label_encoder.pkl"), 'rb') as f:
        le = pickle.load(f)
    with open(os.path.join(base_path, "feature_columns.pkl"), 'rb') as f:
        feature_columns = pickle.load(f)
    with open(os.path.join(base_path, "original_categorical_data.pkl"), 'rb') as f:
        original_categorical_data = pickle.load(f)
        
    return model, le, feature_columns, original_categorical_data

model, le, feature_columns, original_categorical_data = load_model_artifacts()

# --- Streamlit App Layout ---
st.title('Adverse Neonatal Outcome Prediction App')
st.write('Enter the patient details to predict adverse neonatal outcome.')

# --- User Inputs ---
st.sidebar.header('Patient Input Features')
st.write("Available categorical keys:", list(original_categorical_data.keys()))
def user_input_features():
    gdm_status = st.sidebar.selectbox('GDM Status', original_categorical_data['GDM_status'])
    iron_supplementation = st.sidebar.selectbox('Iron Supplementation', original_categorical_data['Ironsupelmentatin'])
    antenatal_depression = st.sidebar.selectbox('Antenatal Depression', original_categorical_data['antenatal_depression'])
    age = st.sidebar.slider('Age', 18, 50, 25)
    residence = st.sidebar.selectbox('Residence', original_categorical_data['Residence'])
    occupation = st.sidebar.selectbox('Occupation', original_categorical_data['Occupation'])
    weight_first_ANC = st.sidebar.slider('Weight at first ANC (kg)', 40.0, 120.0, 65.0, step=0.5)
    muac = st.sidebar.slider('MUAC (cm)', 15.0, 40.0, 25.0, step=0.1)
    bmi = st.sidebar.slider('BMI', 15.0, 40.0, 25.0, step=0.1)
    education = st.sidebar.selectbox('Education', original_categorical_data['Education'])

    data = {
        'Age': age,
        'Residence': residence,
        'Occupation': occupation,
        'weight_first_ANC': weight_first_ANC,
        'MUAC': muac,
        'BMI': bmi,
        'Education': education,
        'GDM_status': gdm_status,
        'Iron_supplementation': iron_supplementation,
        'Antenatal_depression': antenatal_depression
    }
    features = pd.DataFrame(data, index=[0])
    return features

input_df = user_input_features()

st.subheader('User Input Features')
st.write(input_df)

# --- Data Preprocessing for Prediction ---
# Create a DataFrame with all expected columns and fill with zeros
processed_input = pd.DataFrame(0, index=[0], columns=feature_columns)

# Fill numerical features
for col in ['Age', 'weight_first_ANC', 'MUAC', 'BMI']:
    if col in processed_input.columns:
        processed_input[col] = input_df[col].values[0]

# Handle categorical features using one-hot encoding
for col in original_categorical_data.keys():
    if col in input_df.columns:
        val = input_df[col].values[0]
        # Create dummy column name, e.g., 'Residence_Urban'
        dummy_col = f"{col}_{val}"
        if dummy_col in processed_input.columns:
            processed_input[dummy_col] = 1

# Ensure the order of columns matches the training data
processed_input = processed_input[feature_columns]

# --- Prediction ---
prediction = model.predict(processed_input)
prediction_proba = model.predict_proba(processed_input)

st.subheader('Prediction')
predicted_class = le.inverse_transform(prediction)[0]
st.write(predicted_class)

st.subheader('Prediction Probability')
probability_df = pd.DataFrame(prediction_proba, columns=le.classes_)
st.write(probability_df)

st.markdown("""
---
**Note:** This app predicts the likelihood of an adverse neonatal outcome based on the provided inputs.
""")
