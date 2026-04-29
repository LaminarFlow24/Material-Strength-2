import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@st.cache_data
def load_model_and_preprocessors():
    # Load the saved model
    with open('Models/Tens_orientation.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Load training data for UI ranges
    df = pd.read_csv("train1.csv")
    
    # Load saved preprocessors (must match the training environment)
    with open('Models/Tens_encoder.pkl', 'rb') as file:
        encoder = pickle.load(file)
    with open('Models/Tens_scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    with open('Models/Tens_scaler_y.pkl', 'rb') as file:
        scaler_y = pickle.load(file)
    
    return model, encoder, scaler, scaler_y, df

# Load the model and preprocessing objects
model, encoder, scaler, scaler_y, train_df = load_model_and_preprocessors()

st.title("Tensile Strength Prediction App")
st.write("Enter the feature values below to predict the tensile strength of your material.")

st.header("Input Features")

# Categorical inputs
orientation_options = encoder.categories_[0].tolist()
infill_options = encoder.categories_[1].tolist()

orientation = st.selectbox("Orientation", orientation_options)
infill_pattern = st.selectbox("Infill Pattern", infill_options)

# For numeric features, restrict user input within the range available in the CSV
# Layer Thickness
layer_thick_min = float(train_df['layer_thick'].min())
layer_thick_max = float(train_df['layer_thick'].max())
layer_thick_default = float(train_df['layer_thick'].mean())
layer_thick = st.number_input(
    "Layer Thickness", 
    min_value=layer_thick_min, 
    max_value=layer_thick_max, 
    value=layer_thick_default, 
    step=0.01
)

# Infill Density
infill_density_min = float(train_df['infill_density'].min())
infill_density_max = float(train_df['infill_density'].max())
infill_density_default = float(train_df['infill_density'].mean())
infill_density = st.number_input(
    "Infill Density", 
    min_value=infill_density_min, 
    max_value=infill_density_max, 
    value=infill_density_default, 
    step=0.1
)

# MWCNT
mwcnt_min = float(train_df['mwcnt'].min())
mwcnt_max = float(train_df['mwcnt'].max())
mwcnt_default = float(train_df['mwcnt'].mean())
mwcnt = st.number_input(
    "MWCNT", 
    min_value=mwcnt_min, 
    max_value=mwcnt_max, 
    value=mwcnt_default, 
    step=0.1
)

# Graphene
graphene_min = float(train_df['graphene'].min())
graphene_max = float(train_df['graphene'].max())
graphene_default = float(train_df['graphene'].mean())
graphene = st.number_input(
    "Graphene", 
    min_value=graphene_min, 
    max_value=graphene_max, 
    value=graphene_default, 
    step=0.1
)

if st.button("Predict Tensile Strength"):
    # Build a DataFrame with user input
    input_df = pd.DataFrame({
        "orientation": [orientation],
        "infill_pattern": [infill_pattern],
        "layer_thick": [layer_thick],
        "infill_density": [infill_density],
        "mwcnt": [mwcnt],
        "graphene": [graphene]
    })
    
    # Process categorical features using the fitted OneHotEncoder
    input_encoded = encoder.transform(input_df[['orientation', 'infill_pattern']])
    odf_input = pd.DataFrame.sparse.from_spmatrix(input_encoded)
    
    # Get numeric features
    input_numeric = input_df[['layer_thick', 'infill_density', 'mwcnt', 'graphene']]
    
    # Combine encoded categorical features with numeric features
    input_combined = pd.concat([odf_input, input_numeric], axis=1)
    # Add a dummy column for tensile_str to match the scaler's original shape
    input_combined['tensile_str'] = 0
    input_combined.columns = input_combined.columns.astype(str)
    
    # Apply the same StandardScaler used during training
    scaled_input = scaler.transform(input_combined)
    # Remove the dummy target column (assumed to be the last column)
    scaled_input = scaled_input[:, :-1]
    
    # Predict using the loaded model
    pred_scaled = model.predict(scaled_input)
    # Inverse-transform the prediction back to the original scale
    pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    st.success(f"Predicted Tensile Strength: {pred_original[0,0]:.2f}")
