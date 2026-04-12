import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler

@st.cache(allow_output_mutation=True)
def load_model_and_preprocessors():
    # Load your saved flexural strength model
    with open('Models/Flex_orientation.pkl', 'rb') as file:
        model = pickle.load(file)
    
    # Load training data to re-fit preprocessors and extract ranges
    df = pd.read_csv("train1.csv")
    
    # Fit OneHotEncoder on the categorical features
    encoder = OneHotEncoder()
    encoder.fit(df[['orientation', 'infill_pattern']])
    
    # Create a combined DataFrame (mimicking training) for scaler fitting
    df_encoded = encoder.transform(df[['orientation','infill_pattern']])
    odf = pd.DataFrame.sparse.from_spmatrix(df_encoded)
    
    # Define numeric columns used during training (including the target)
    numeric_cols = ['layer_thick', 'infill_density', 'mwcnt', 'graphene', 'flexural_str']
    idf = df[numeric_cols]
    
    # Combine the one-hot encoded columns with numeric columns
    cdf = pd.concat([odf, idf], axis=1)
    cdf.columns = cdf.columns.astype(str)
    
    # Fit the scaler on the entire DataFrame (including target)
    scaler = StandardScaler()
    scaler.fit(cdf)
    
    # Fit a separate scaler for the target so we can inverse-transform predictions
    scaler_y = StandardScaler()
    scaler_y.fit(df[['flexural_str']])
    
    return model, encoder, scaler, scaler_y, df

# Load model and preprocessors
model, encoder, scaler, scaler_y, train_df = load_model_and_preprocessors()

st.title("Flexural Strength vs. Varying Parameter with Noise")
st.write("""
Select one parameter to vary and provide fixed values for the remaining five.
The app will generate a graph showing the predicted flexural strength (using your model) 
across the available values in the CSV for the chosen variable.
""")

# List of all input parameters
params = ["orientation", "infill_pattern", "layer_thick", "infill_density", "mwcnt", "graphene"]

# Let the user select which parameter to vary
var_param = st.selectbox("Select the parameter to vary:", params)

# The remaining parameters will be fixed by user input
fixed_params = [p for p in params if p != var_param]

st.subheader("Enter fixed parameter values:")

fixed_values = {}

# For each fixed parameter, display an input widget with range restrictions for numeric ones.
for param in fixed_params:
    if param in ["orientation", "infill_pattern"]:
        # For categorical parameters, get options from the fitted encoder
        if param == "orientation":
            options = encoder.categories_[0].tolist()
        else:
            options = encoder.categories_[1].tolist()
        fixed_values[param] = st.selectbox(f"Select value for {param}:", options)
    else:
        # For numeric parameters, restrict input to the CSV's range.
        param_min = float(train_df[param].min())
        param_max = float(train_df[param].max())
        param_default = float(train_df[param].mean())
        # Use a step size based on the parameter (adjust if needed)
        step_val = 0.01 if param == "layer_thick" else 0.1
        fixed_values[param] = st.number_input(
            f"Enter value for {param}:",
            min_value=param_min,
            max_value=param_max,
            value=param_default,
            step=step_val
        )

if st.button("Generate Graph"):
    # Determine values for the variable parameter.
    if var_param not in ["orientation", "infill_pattern"]:
        # Numeric parameter: use unique sorted values from training data.
        var_range = np.sort(train_df[var_param].unique())
    else:
        # Categorical parameter: use all categories from the fitted encoder.
        if var_param == "orientation":
            var_range = encoder.categories_[0].tolist()
        else:
            var_range = encoder.categories_[1].tolist()
    
    # Build a DataFrame with one row per candidate value of the varying parameter.
    # The fixed parameters are repeated.
    data = {}
    for param in params:
        if param == var_param:
            data[param] = var_range
        else:
            data[param] = [fixed_values[param]] * len(var_range)
    
    input_df = pd.DataFrame(data)
    
    # Preprocess the input DataFrame as during training:
    # 1. One-hot encode the categorical features.
    input_encoded = encoder.transform(input_df[['orientation', 'infill_pattern']])
    odf_input = pd.DataFrame.sparse.from_spmatrix(input_encoded)
    
    # 2. Process numeric features.
    numeric_cols = ['layer_thick', 'infill_density', 'mwcnt', 'graphene']
    idf_input = input_df[numeric_cols].copy()
    # Add a dummy target column (flexural_str) to match the scaler's input shape.
    idf_input['flexural_str'] = 0
    
    # 3. Combine encoded categorical data with numeric data.
    combined_input = pd.concat([odf_input, idf_input], axis=1)
    combined_input.columns = combined_input.columns.astype(str)
    
    # 4. Scale the input data using the fitted scaler.
    scaled_input = scaler.transform(combined_input)
    # Remove the dummy target column (assumed to be the last column).
    scaled_input = scaled_input[:, :-1]
    
    # Use the model to predict flexural strength on the scaled input.
    pred_scaled = model.predict(scaled_input)
    # Inverse-transform the prediction back to the original scale.
    pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    # Add noise to predictions to create a more natural-looking graph.
    noise_std = 0.05 * np.std(pred_original)
    noise = np.random.normal(0, noise_std, pred_original.shape)
    pred_noisy = pred_original + noise
    
    # Plot the predictions.
    fig, ax = plt.subplots()
    if var_param not in ["orientation", "infill_pattern"]:
        # For numeric variable, plot a line chart.
        ax.plot(var_range, pred_noisy, marker='o', linestyle='-')
        ax.set_xlabel(var_param)
        ax.set_ylabel("Predicted Flexural Strength")
        ax.set_title(f"Flexural Strength vs. {var_param}")
    else:
        # For categorical variable, display a bar chart.
        ax.bar(var_range, pred_noisy.flatten())
        ax.set_xlabel(var_param)
        ax.set_ylabel("Predicted Flexural Strength")
        ax.set_title(f"Flexural Strength vs. {var_param}")
    
    st.pyplot(fig)
