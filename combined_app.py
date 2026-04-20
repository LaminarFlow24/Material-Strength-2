import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import requests
import json
import os
from dotenv import load_dotenv

load_dotenv()

# Configure page
st.set_page_config(
    page_title="Material Strength Prediction Suite",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

@st.cache_data
def load_tensile_model_and_preprocessors():
    """Load tensile strength model and preprocessing objects"""
    try:
        with open('Models/Tens_orientation.pkl', 'rb') as file:
            model = pickle.load(file)
        
        df = pd.read_csv("train1.csv")
        
        encoder = OneHotEncoder()
        encoder.fit(df[['orientation', 'infill_pattern']])
        
        df_encoded = encoder.transform(df[['orientation','infill_pattern']])
        odf = pd.DataFrame.sparse.from_spmatrix(df_encoded)
        
        numeric_cols = ['layer_thick', 'infill_density', 'mwcnt', 'graphene', 'tensile_str']
        idf = df[numeric_cols]
        
        cdf = pd.concat([odf, idf], axis=1)
        cdf.columns = cdf.columns.astype(str)
        
        scaler = StandardScaler()
        scaler.fit(cdf)
        
        scaler_y = StandardScaler()
        scaler_y.fit(df[['tensile_str']])
        
        return model, encoder, scaler, scaler_y, df
    except Exception as e:
        st.error(f"Error loading tensile model: {e}")
        return None, None, None, None, None

@st.cache_data
def load_flexural_model_and_preprocessors():
    """Load flexural strength model and preprocessing objects"""
    try:
        with open('Models/Flex_orientation.pkl', 'rb') as file:
            model = pickle.load(file)
        
        df = pd.read_csv("train1.csv")
        
        encoder = OneHotEncoder()
        encoder.fit(df[['orientation', 'infill_pattern']])
        
        df_encoded = encoder.transform(df[['orientation','infill_pattern']])
        odf = pd.DataFrame.sparse.from_spmatrix(df_encoded)
        
        numeric_cols = ['layer_thick', 'infill_density', 'mwcnt', 'graphene', 'flexural_str']
        idf = df[numeric_cols]
        
        cdf = pd.concat([odf, idf], axis=1)
        cdf.columns = cdf.columns.astype(str)
        
        scaler = StandardScaler()
        scaler.fit(cdf)
        
        scaler_y = StandardScaler()
        scaler_y.fit(df[['flexural_str']])
        
        return model, encoder, scaler, scaler_y, df
    except Exception as e:
        st.error(f"Error loading flexural model: {e}")
        return None, None, None, None, None

def get_input_parameters(train_df, key_suffix=""):
    """Get input parameters from user with unique keys"""
    # Get options from the training data
    orientation_options = sorted(train_df['orientation'].unique())
    infill_options = sorted(train_df['infill_pattern'].unique())
    
    col1, col2 = st.columns(2)
    
    with col1:
        orientation = st.selectbox("Orientation", orientation_options, key=f"orientation_{key_suffix}")
        infill_pattern = st.selectbox("Infill Pattern", infill_options, key=f"infill_{key_suffix}")
        
        layer_thick_min = float(train_df['layer_thick'].min())
        layer_thick_max = float(train_df['layer_thick'].max())
        layer_thick_default = float(train_df['layer_thick'].mean())
        layer_thick = st.number_input(
            "Layer Thickness", 
            min_value=layer_thick_min, 
            max_value=layer_thick_max, 
            value=layer_thick_default, 
            step=0.01,
            key=f"layer_thick_{key_suffix}"
        )
    
    with col2:
        infill_density_min = float(train_df['infill_density'].min())
        infill_density_max = float(train_df['infill_density'].max())
        infill_density_default = float(train_df['infill_density'].mean())
        infill_density = st.number_input(
            "Infill Density", 
            min_value=infill_density_min, 
            max_value=infill_density_max, 
            value=infill_density_default, 
            step=0.1,
            key=f"infill_density_{key_suffix}"
        )
        
        mwcnt_min = float(train_df['mwcnt'].min())
        mwcnt_max = float(train_df['mwcnt'].max())
        mwcnt_default = float(train_df['mwcnt'].mean())
        mwcnt = st.number_input(
            "MWCNT", 
            min_value=mwcnt_min, 
            max_value=mwcnt_max, 
            value=mwcnt_default, 
            step=0.1,
            key=f"mwcnt_{key_suffix}"
        )
        
        graphene_min = float(train_df['graphene'].min())
        graphene_max = float(train_df['graphene'].max())
        graphene_default = float(train_df['graphene'].mean())
        graphene = st.number_input(
            "Graphene", 
            min_value=graphene_min, 
            max_value=graphene_max, 
            value=graphene_default, 
            step=0.1,
            key=f"graphene_{key_suffix}"
        )
    
    return {
        "orientation": orientation,
        "infill_pattern": infill_pattern,
        "layer_thick": layer_thick,
        "infill_density": infill_density,
        "mwcnt": mwcnt,
        "graphene": graphene
    }

def make_prediction(input_params, model, encoder, scaler, scaler_y, target_col):
    """Make prediction using the provided model and preprocessors"""
    try:
        # Build a DataFrame with user input
        input_df = pd.DataFrame({
            "orientation": [input_params["orientation"]],
            "infill_pattern": [input_params["infill_pattern"]],
            "layer_thick": [input_params["layer_thick"]],
            "infill_density": [input_params["infill_density"]],
            "mwcnt": [input_params["mwcnt"]],
            "graphene": [input_params["graphene"]]
        })
        
        # Process categorical features using the fitted OneHotEncoder
        input_encoded = encoder.transform(input_df[['orientation', 'infill_pattern']])
        odf_input = pd.DataFrame.sparse.from_spmatrix(input_encoded)
        
        # Get numeric features
        input_numeric = input_df[['layer_thick', 'infill_density', 'mwcnt', 'graphene']]
        
        # Combine encoded categorical features with numeric features
        input_combined = pd.concat([odf_input, input_numeric], axis=1)
        # Add a dummy column for target to match the scaler's original shape
        input_combined[target_col] = 0
        input_combined.columns = input_combined.columns.astype(str)
        
        # Apply the same StandardScaler used during training
        scaled_input = scaler.transform(input_combined)
        # Remove the dummy target column (assumed to be the last column)
        scaled_input = scaled_input[:, :-1]
        
        # Predict using the loaded model
        pred_scaled = model.predict(scaled_input)
        # Inverse-transform the prediction back to the original scale
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        
        return pred_original[0,0]
    except Exception as e:
        st.error(f"Error making prediction: {e}")
        return None

def get_gemini_insights(prediction, params, strength_type):
    """Get materials science insights from Gemini"""
    api_key = os.environ.get("GEMINI_API_KEY")
    prompt = f"Act as an expert in 3D printing and materials science. I have printed a composite material with the following parameters: " \
             f"Orientation: {params['orientation']}, Infill Pattern: {params['infill_pattern']}, " \
             f"Layer Thickness: {params['layer_thick']} mm, Infill Density: {params['infill_density']} %, " \
             f"MWCNT: {params['mwcnt']} %, Graphene: {params['graphene']} %. " \
             f"This resulted in a predicted {strength_type} strength of {prediction:.2f} MPa. " \
             f"Based on these parameters, provide point-wise reasoning (3-4 concise points) explaining why it achieved this strength " \
             f"and what microscopic mechanics might be at play. " \
             f"CRITICAL: Always include appropriate units (e.g., mm, %, MPa) immediately after any numerical values in your response."
             
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    headers = {'Content-Type': 'application/json'}
    data = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    
    try:
        response = requests.post(url, headers=headers, json=data)
        if response.status_code == 200:
            return response.json()['candidates'][0]['content']['parts'][0]['text']
        else:
            raise Exception(f"Gemini API Error: {response.text}")
    except Exception as e:
        # Fallback to Grok/Groq API
        fallback_key = os.environ.get("GROK_API_KEY")
        if not fallback_key:
            return f"Error connecting to Gemini AI: {e} (No fallback key available)"
        
        try:
            fallback_url = "https://api.groq.com/openai/v1/chat/completions"
            fallback_headers = {
                "Authorization": f"Bearer {fallback_key}",
                "Content-Type": "application/json"
            }
            fallback_data = {
                "model": "openai/gpt-oss-120b",
                "messages": [{"role": "user", "content": prompt}]
            }
            fb_response = requests.post(fallback_url, headers=fallback_headers, json=fallback_data)
            
            if fb_response.status_code == 200:
                return fb_response.json()['choices'][0]['message']['content']
            else:
                return f"Gemini Error: {e}\nFallback API Error: {fb_response.text}"
        except Exception as fb_e:
            return f"Gemini Error: {e}\nFallback API Exception: {fb_e}"

def find_optimal_conditions(model, encoder, scaler, scaler_y, train_df, target_col):
    """Find parameters yielding the highest predicted strength"""
    orientations = sorted(train_df['orientation'].unique())
    infills = sorted(train_df['infill_pattern'].unique())
    layer_ticks = train_df['layer_thick'].unique()
    infill_densities = train_df['infill_density'].unique()
    mwcnts = train_df['mwcnt'].unique()
    graphenes = train_df['graphene'].unique()
    
    data = []
    for o in orientations:
        for i in infills:
            for l in layer_ticks:
                for ind in infill_densities:
                    for m in mwcnts:
                        for g in graphenes:
                            if m > 1.5 or g > 1.5:
                                continue # Individual nanocomposite constraint
                            data.append({'orientation': o, 'infill_pattern': i, 'layer_thick': l, 
                                        'infill_density': ind, 'mwcnt': m, 'graphene': g})
    
    if not data:
        return None, None
        
    calc_df = pd.DataFrame(data)
    
    input_encoded = encoder.transform(calc_df[['orientation', 'infill_pattern']])
    odf_input = pd.DataFrame.sparse.from_spmatrix(input_encoded)
    
    input_numeric = calc_df[['layer_thick', 'infill_density', 'mwcnt', 'graphene']]
    
    combined_input = pd.concat([odf_input, input_numeric], axis=1)
    combined_input[target_col] = 0
    combined_input.columns = combined_input.columns.astype(str)
    
    scaled_input = scaler.transform(combined_input)
    scaled_input = scaled_input[:, :-1]
    
    pred_scaled = model.predict(scaled_input)
    pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
    
    max_idx = np.argmax(pred_original)
    return pred_original[max_idx][0], data[max_idx]

def tensile_prediction_page():
    """Tensile strength prediction page"""
    st.title("🔧 Tensile Strength Prediction")
    st.write("Enter the feature values below to predict the tensile strength of your material.")
    
    # Load model
    model, encoder, scaler, scaler_y, train_df = load_tensile_model_and_preprocessors()
    
    if model is None:
        st.error("Failed to load tensile strength model. Please check if the model file exists.")
        return
    
    st.header("Input Features")
    input_params = get_input_parameters(train_df, "tensile")
    
    if st.button("Predict Tensile Strength", type="primary"):
        if input_params["mwcnt"] > 1.5 or input_params["graphene"] > 1.5:
            st.error("Constraint Failed: MWCNT and Graphene cannot individually exceed 1.5%.")
        else:
            prediction = make_prediction(input_params, model, encoder, scaler, scaler_y, "tensile_str")
            if prediction is not None:
                st.success(f"🎯 **Predicted Tensile Strength: {prediction:.2f}**")
                
                # Display input summary
                with st.expander("Input Summary"):
                    st.json(input_params)
                
                with st.spinner("Generating AI Analysis..."):
                    ai_insight = get_gemini_insights(prediction, input_params, "Tensile")
                st.info(f"🤖 **Gemini AI Insight:**\n\n{ai_insight}")
                
    st.markdown("---")
    st.subheader("Optimal Conditions")
    st.write("Find the realistic parameters that yield the absolute highest tensile strength.")
    if st.button("Discover Optimal Parameters for Tensile"):
        best_pred, best_params = find_optimal_conditions(model, encoder, scaler, scaler_y, train_df, "tensile_str")
        if best_params:
            st.success(f"🏆 **Maximum Achievable Tensile Strength: {best_pred:.2f} MPa**")
            st.json(best_params)
            
            with st.spinner("Generating AI Analysis..."):
                ai_insight = get_gemini_insights(best_pred, best_params, "Tensile")
            st.info(f"🤖 **Gemini AI Insight on Optimal Parameters:**\n\n{ai_insight}")

def flexural_prediction_page():
    """Flexural strength prediction page"""
    st.title("🔩 Flexural Strength Prediction")
    st.write("Enter the feature values below to predict the flexural strength of your material.")
    
    # Load model
    model, encoder, scaler, scaler_y, train_df = load_flexural_model_and_preprocessors()
    
    if model is None:
        st.error("Failed to load flexural strength model. Please check if the model file exists.")
        return
    
    st.header("Input Features")
    input_params = get_input_parameters(train_df, "flexural")
    
    if st.button("Predict Flexural Strength", type="primary"):
        if input_params["mwcnt"] > 1.5 or input_params["graphene"] > 1.5:
            st.error("Constraint Failed: MWCNT and Graphene cannot individually exceed 1.5%.")
        else:
            prediction = make_prediction(input_params, model, encoder, scaler, scaler_y, "flexural_str")
            if prediction is not None:
                st.success(f"🎯 **Predicted Flexural Strength: {prediction:.2f}**")
                
                # Display input summary
                with st.expander("Input Summary"):
                    st.json(input_params)
                
                with st.spinner("Generating AI Analysis..."):
                    ai_insight = get_gemini_insights(prediction, input_params, "Flexural")
                st.info(f"🤖 **Gemini AI Insight:**\n\n{ai_insight}")
                
    st.markdown("---")
    st.subheader("Optimal Conditions")
    st.write("Find the realistic parameters that yield the absolute highest flexural strength.")
    if st.button("Discover Optimal Parameters for Flexural"):
        best_pred, best_params = find_optimal_conditions(model, encoder, scaler, scaler_y, train_df, "flexural_str")
        if best_params:
            st.success(f"🏆 **Maximum Achievable Flexural Strength: {best_pred:.2f} MPa**")
            st.json(best_params)
            
            with st.spinner("Generating AI Analysis..."):
                ai_insight = get_gemini_insights(best_pred, best_params, "Flexural")
            st.info(f"🤖 **Gemini AI Insight on Optimal Parameters:**\n\n{ai_insight}")

def generate_graph(model, encoder, scaler, scaler_y, train_df, var_param, fixed_values, target_col, strength_type):
    """Generate graph for varying parameter"""
    try:
        # List of all input parameters
        params = ["orientation", "infill_pattern", "layer_thick", "infill_density", "mwcnt", "graphene"]
        
        # Determine values for the variable parameter
        if var_param not in ["orientation", "infill_pattern"]:
            var_range = np.sort(train_df[var_param].unique())
        else:
            if var_param == "orientation":
                var_range = sorted(train_df['orientation'].unique())
            else:
                var_range = sorted(train_df['infill_pattern'].unique())
        
        # Build a DataFrame with one row per candidate value of the varying parameter
        data = {}
        for param in params:
            if param == var_param:
                data[param] = var_range
            else:
                data[param] = [fixed_values[param]] * len(var_range)
        
        input_df = pd.DataFrame(data)
        
        # Preprocess the input DataFrame as during training
        input_encoded = encoder.transform(input_df[['orientation', 'infill_pattern']])
        odf_input = pd.DataFrame.sparse.from_spmatrix(input_encoded)
        
        numeric_cols = ['layer_thick', 'infill_density', 'mwcnt', 'graphene']
        idf_input = input_df[numeric_cols].copy()
        idf_input[target_col] = 0
        
        combined_input = pd.concat([odf_input, idf_input], axis=1)
        combined_input.columns = combined_input.columns.astype(str)
        
        scaled_input = scaler.transform(combined_input)
        scaled_input = scaled_input[:, :-1]
        
        pred_scaled = model.predict(scaled_input)
        pred_original = scaler_y.inverse_transform(pred_scaled.reshape(-1, 1))
        
        # Add noise to predictions
        noise_std = 0.05 * np.std(pred_original)
        noise = np.random.normal(0, noise_std, pred_original.shape)
        pred_noisy = pred_original + noise
        
        # Plot the predictions
        fig, ax = plt.subplots(figsize=(10, 6))
        if var_param not in ["orientation", "infill_pattern"]:
            ax.plot(var_range, pred_noisy, marker='o', linestyle='-', linewidth=2, markersize=8)
            ax.set_xlabel(var_param.replace('_', ' ').title())
            ax.set_ylabel(f"Predicted {strength_type} Strength")
            ax.set_title(f"{strength_type} Strength vs. {var_param.replace('_', ' ').title()}")
        else:
            bars = ax.bar(var_range, pred_noisy.flatten())
            ax.set_xlabel(var_param.replace('_', ' ').title())
            ax.set_ylabel(f"Predicted {strength_type} Strength")
            ax.set_title(f"{strength_type} Strength vs. {var_param.replace('_', ' ').title()}")
            
            # Add value labels on bars
            for bar, value in zip(bars, pred_noisy.flatten()):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                       f'{value:.1f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error generating graph: {e}")
        return None

def tensile_graph_page():
    """Tensile strength graphing page"""
    st.title("📊 Tensile Strength Analysis")
    st.write("""
    Select one parameter to vary and provide fixed values for the remaining parameters.
    The app will generate a graph showing the predicted tensile strength across different values.
    """)
    
    # Load model
    model, encoder, scaler, scaler_y, train_df = load_tensile_model_and_preprocessors()
    
    if model is None:
        st.error("Failed to load tensile strength model. Please check if the model file exists.")
        return
    
    params = ["orientation", "infill_pattern", "layer_thick", "infill_density", "mwcnt", "graphene"]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Graph Configuration")
        var_param = st.selectbox("Parameter to vary:", params, key="tensile_var_param")
        
        fixed_params = [p for p in params if p != var_param]
        
        st.subheader("Fixed parameter values:")
        fixed_values = {}
        
        for param in fixed_params:
            if param in ["orientation", "infill_pattern"]:
                if param == "orientation":
                    options = sorted(train_df['orientation'].unique())
                else:
                    options = sorted(train_df['infill_pattern'].unique())
                fixed_values[param] = st.selectbox(f"{param.replace('_', ' ').title()}:", options, key=f"tensile_fixed_{param}")
            else:
                param_min = float(train_df[param].min())
                param_max = float(train_df[param].max())
                param_default = float(train_df[param].mean())
                step_val = 0.01 if param == "layer_thick" else 0.1
                fixed_values[param] = st.number_input(
                    f"{param.replace('_', ' ').title()}:",
                    min_value=param_min,
                    max_value=param_max,
                    value=param_default,
                    step=step_val,
                    key=f"tensile_fixed_{param}"
                )
    
    with col2:
        if st.button("Generate Tensile Strength Graph", type="primary"):
            fig = generate_graph(model, encoder, scaler, scaler_y, train_df, var_param, fixed_values, "tensile_str", "Tensile")
            if fig is not None:
                st.pyplot(fig)

def flexural_graph_page():
    """Flexural strength graphing page"""
    st.title("📈 Flexural Strength Analysis")
    st.write("""
    Select one parameter to vary and provide fixed values for the remaining parameters.
    The app will generate a graph showing the predicted flexural strength across different values.
    """)
    
    # Load model
    model, encoder, scaler, scaler_y, train_df = load_flexural_model_and_preprocessors()
    
    if model is None:
        st.error("Failed to load flexural strength model. Please check if the model file exists.")
        return
    
    params = ["orientation", "infill_pattern", "layer_thick", "infill_density", "mwcnt", "graphene"]
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Graph Configuration")
        var_param = st.selectbox("Parameter to vary:", params, key="flexural_var_param")
        
        fixed_params = [p for p in params if p != var_param]
        
        st.subheader("Fixed parameter values:")
        fixed_values = {}
        
        for param in fixed_params:
            if param in ["orientation", "infill_pattern"]:
                if param == "orientation":
                    options = sorted(train_df['orientation'].unique())
                else:
                    options = sorted(train_df['infill_pattern'].unique())
                fixed_values[param] = st.selectbox(f"{param.replace('_', ' ').title()}:", options, key=f"flexural_fixed_{param}")
            else:
                param_min = float(train_df[param].min())
                param_max = float(train_df[param].max())
                param_default = float(train_df[param].mean())
                step_val = 0.01 if param == "layer_thick" else 0.1
                fixed_values[param] = st.number_input(
                    f"{param.replace('_', ' ').title()}:",
                    min_value=param_min,
                    max_value=param_max,
                    value=param_default,
                    step=step_val,
                    key=f"flexural_fixed_{param}"
                )
    
    with col2:
        if st.button("Generate Flexural Strength Graph", type="primary"):
            fig = generate_graph(model, encoder, scaler, scaler_y, train_df, var_param, fixed_values, "flexural_str", "Flexural")
            if fig is not None:
                st.pyplot(fig)

def main():
    """Main application"""
    # Sidebar navigation
    st.sidebar.title("🔬 Material Strength Prediction Suite")
    st.sidebar.markdown("---")
    
    page = st.sidebar.selectbox(
        "Choose Analysis Type:",
        [
            "🔧 Tensile Strength Prediction",
            "🔩 Flexural Strength Prediction", 
            "📊 Tensile Strength Analysis",
            "📈 Flexural Strength Analysis"
        ]
    )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("""
    ### About This App
    This comprehensive tool provides:
    - **Prediction**: Get strength values for specific material parameters
    - **Analysis**: Visualize how parameters affect material strength
    - **Models**: Both tensile and flexural strength prediction
    
    ### How to Use
    1. Select the analysis type from the sidebar
    2. Enter your material parameters
    3. Get predictions or generate graphs
    """)
    
    # Route to appropriate page
    if page == "🔧 Tensile Strength Prediction":
        tensile_prediction_page()
    elif page == "🔩 Flexural Strength Prediction":
        flexural_prediction_page()
    elif page == "📊 Tensile Strength Analysis":
        tensile_graph_page()
    elif page == "📈 Flexural Strength Analysis":
        flexural_graph_page()

if __name__ == "__main__":
    main()