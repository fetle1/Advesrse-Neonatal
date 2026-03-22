import streamlit as st
import pandas as pd
import joblib
import pickle
import os

# --- Page Config ---
st.set_page_config(
    page_title="Neonatal Outcome Predictor",
    layout="centered"
)
st.markdown("""
<div style='background-color:white; padding:10px; border-radius:15px; box-shadow:0 2px 8px rgba(0,0,0,0.1);'>
""", unsafe_allow_html=True)

st.image(
    "https://assets.clevelandclinic.org/transform/LargeFeatureImage/21510f8a-8787-45c0-b07d-e67a81c9c17b/mom-baby-clo",
    use_container_width=True
)

st.markdown("</div>", unsafe_allow_html=True)
# --- Custom CSS Styling ---
st.markdown("""
<style>

/* Full app background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(to right, #e3f2fd, #fce4ec);
}

/* Main content block */
[data-testid="stMain"] {
    background: transparent;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #f8bbd0;
}

/* Buttons */
.stButton>button {
    background-color: #64b5f6;
    color: white;
    border-radius: 12px;
    height: 3em;
    width: 100%;
    font-size: 16px;
    border: none;
}

/* Result box */
.result-box {
    padding: 20px;
    border-radius: 15px;
    background-color: #ffffff;
    text-align: center;
    font-size: 22px;
    font-weight: bold;
    animation: fadeIn 1.5s ease-in-out;
    box-shadow: 0 4px 10px rgba(0,0,0,0.1);
}

/* Animation */
@keyframes fadeIn {
    from {opacity: 0; transform: translateY(10px);}
    to {opacity: 1; transform: translateY(0);}
}

</style>
""", unsafe_allow_html=True)
# --- Load Model ---
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

# Fix NaNs
for key, values in original_categorical_data.items():
    original_categorical_data[key] = [v if pd.notna(v) else "Unknown" for v in values]

# --- Header Section ---
st.markdown("<h1 style='text-align: center;'>👶 Adverse Neonatal Outcome Predictor</h1>", unsafe_allow_html=True)



st.markdown("<p style='text-align:center;'>Enter maternal details to assess neonatal risk</p>", unsafe_allow_html=True)

# --- Sidebar Input ---
st.sidebar.header('📝 Patient Information')

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
    return pd.DataFrame(data, index=[0])

input_df = user_input_features()

# --- Show Inputs in Card ---
st.markdown("### 📋 Patient Summary")
st.dataframe(input_df, use_container_width=True)

# --- Preprocessing ---
processed_input = pd.DataFrame(0, index=[0], columns=feature_columns)

for col in ['Age', 'weight_first_ANC', 'MUAC', 'BMI']:
    if col in processed_input.columns:
        processed_input[col] = input_df[col].values[0]

for col in original_categorical_data.keys():
    if col in input_df.columns:
        val = input_df[col].values[0]
        dummy_col = f"{col}_{val}"
        if dummy_col in processed_input.columns:
            processed_input[dummy_col] = 1

processed_input = processed_input[feature_columns]

# --- Predict Button ---
if st.button("🔍 Predict Outcome"):

    prediction = model.predict(processed_input)
    prediction_proba = model.predict_proba(processed_input)

    predicted_class = le.inverse_transform(prediction)[0]

    st.markdown("## 🎯 Prediction Result")

    # Animated Result Box
    if predicted_class.lower() == "adverse":
        st.markdown(f"<div class='result-box'>⚠️ High Risk: {predicted_class}</div>", unsafe_allow_html=True)
        st.error("This indicates a higher likelihood of adverse neonatal outcome.")
    else:
        st.markdown(f"<div class='result-box'>✅ Low Risk: {predicted_class}</div>", unsafe_allow_html=True)
        st.success("This indicates a lower likelihood of adverse neonatal outcome.")

    # Probability Bar
    st.markdown("### 📊 Prediction Probability")
    prob_df = pd.DataFrame(prediction_proba, columns=le.classes_)
    st.bar_chart(prob_df.T)

# --- Footer ---
st.markdown("""
---
💡 *This tool supports clinical decision-making but does not replace professional judgment.*
""")
