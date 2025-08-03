import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
import re
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# ==============================================================================
# APPLICATION CONFIGURATION
# ==============================================================================
st.set_page_config(page_title="Weather & Crop Prediction", layout="centered")
st.title("🌦️🌱 Weather Forecast & Crop Recommendation")

# ==============================================================================
# PATH DEFINITIONS AND CONSTANTS
# ==============================================================================
MODEL_FOLDER = 'models/'
SCALER_LSTM_FOLDER = 'scaler_lstm/'
DATA_FILE = 'climate_data_with_province.csv'
RF_CROP_MODEL_FILE = "rf_crop_model.pkl"
CROP_SCALER_FILE = "crop_scaler.pkl"
LABEL_ENCODER_FILE = "label_encoder_crop.pkl"

# ==============================================================================
# HELPER FUNCTIONS (WITH CACHING)
# ==============================================================================

def get_available_provinces(model_folder):
    """Scans the model folder to get a list of available provinces."""
    provinces = set()
    if not os.path.exists(model_folder):
        return []
    
    for filename in os.listdir(model_folder):
        if filename.endswith(".h5"):
            # Remove the '.h5' extension from the filename
            name_part = filename[:-3] # Removing '.h5'
            
            # Remove the known prefix
            if name_part.startswith('finetuned_'):
                province_name_with_underscores = name_part.replace('finetuned_', '', 1)
            elif name_part.startswith('lstm_attention_'):
                province_name_with_underscores = name_part.replace('lstm_attention_', '', 1)
            else:
                # If another format is found, just skip it
                continue
                
            # Replace underscores with spaces and add to the set
            provinces.add(province_name_with_underscores.replace('_', ' '))
            
    return sorted(list(provinces))

@st.cache_data
def load_climate_data(file_path):
    """Loads historical climate data from a CSV file."""
    if not os.path.exists(file_path):
        st.error(f"Data file '{file_path}' not found.")
        st.stop()
    return pd.read_csv(file_path)

@st.cache_resource
def load_crop_models():
    """Loads all models required for crop recommendation."""
    try:
        rf_model = joblib.load(RF_CROP_MODEL_FILE)
        scaler = joblib.load(CROP_SCALER_FILE)
        encoder = joblib.load(LABEL_ENCODER_FILE)
        return rf_model, scaler, encoder
    except FileNotFoundError as e:
        st.error(f"Failed to load crop model file: {e.name}. Make sure the .pkl files are in the correct directory.")
        st.stop()

@st.cache_resource
def load_weather_model(model_path):
    """Loads a Keras weather model from the given path."""
    return load_model(model_path, compile=False)

# ==============================================================================
# LOADING STATIC DATA & MODELS
# ==============================================================================
climate_data = load_climate_data(DATA_FILE)
rf_crop_model, crop_scaler, label_encoder = load_crop_models()

features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss']
targets = ['Tavg', 'RH_avg', 'RR']

# ==============================================================================
# USER INPUT
# ==============================================================================
st.header("👨‍🌾 Enter Your Land Information")
col1, col2 = st.columns(2)
with col1:
    n = st.number_input("Nitrogen Content (N)", min_value=0, value=0)
    p = st.number_input("Phosphorus Content (P)", min_value=0, value=0)
    k = st.number_input("Potassium Content (K)", min_value=0, value=0)
with col2:
    ph = st.number_input("Soil pH Level", min_value=0.0, max_value=14.0, value=0.0, format="%.1f")
    provinsi_list = get_available_provinces(MODEL_FOLDER)
    if not provinsi_list:
        st.error(f"No models found in the '{MODEL_FOLDER}' folder. The application cannot continue.")
        st.stop()
    provinsi = st.selectbox("Select Location (Province)", provinsi_list)
    
jumlah_hari_prediksi = st.slider("Select Number of Prediction Days", min_value=30, max_value=60, value=60)

# Button to start the prediction
if st.button("Generate Prediction and Recommendation", type="primary"):

    # ==============================================================================
    # LONG-TERM WEATHER PREDICTION PROCESS
    # ==============================================================================
    st.header(f"Weather Forecast for the Next {jumlah_hari_prediksi} Days")

    # --- 1. Filter and validate historical data ---
    df_prov = climate_data[climate_data['province_name'] == provinsi].dropna(subset=features)
    if len(df_prov) < 30:
        st.error(f"Not enough data for province '{provinsi}' for prediction (less than 30 days).")
        st.stop()

    # --- 2. Load the province-specific weather model and scaler ---
    try:
        scaler_path = os.path.join(SCALER_LSTM_FOLDER, f"scaler_lstm_{provinsi}.pkl")
        scaler = joblib.load(scaler_path)
    except FileNotFoundError:
        st.error(f"Weather scaler for '{provinsi}' not found at '{scaler_path}'.")
        st.stop()

    path_finetuned = os.path.join(MODEL_FOLDER, f'finetuned_{provinsi}.h5')
    path_lstm_attention = os.path.join(MODEL_FOLDER, f'lstm_attention_{provinsi}.h5')
    
    if os.path.exists(path_finetuned):
        model_path = path_finetuned
    elif os.path.exists(path_lstm_attention):
        model_path = path_lstm_attention
    else:
        st.error(f"Weather model for province '{provinsi}' not found in the '{MODEL_FOLDER}' folder.")
        st.stop()
        
    st.caption(f"Using model: `{os.path.basename(model_path)}`")
    model = load_weather_model(model_path)

    # --- 3. Prepare initial input for the LSTM ---
    df_input = df_prov.tail(30).copy()
    input_scaled = scaler.transform(df_input[features])
    X_input = input_scaled.reshape(1, 30, len(features))

    # --- 4. Perform Iterative Prediction ---
    list_prediksi = []
    with st.spinner(f"Generating predictions for {jumlah_hari_prediksi} days..."):
        for _ in range(jumlah_hari_prediksi):
            # Predict 1 step ahead
            pred_norm = model.predict(X_input, verbose=0)[0]
            
            # Create a dummy array with the size of the input features, then fill it with the predicted values
            dummy_pred_full_features = np.zeros((1, len(features)))
            dummy_pred_full_features[0, features.index('Tavg')] = pred_norm[0]
            dummy_pred_full_features[0, features.index('RH_avg')] = pred_norm[1]
            dummy_pred_full_features[0, features.index('RR')] = pred_norm[2]
            
            # Inverse transform the prediction to get the original values
            pred_denorm = scaler.inverse_transform(dummy_pred_full_features)[0]
            list_prediksi.append([
                pred_denorm[features.index('Tavg')],
                pred_denorm[features.index('RH_avg')],
                pred_denorm[features.index('RR')]
            ])

            # Create the new input for the next iteration
            # Combine the new prediction with other features from the previous time step
            new_input_scaled = X_input[0, -1, :].copy() # Take features from the last day
            new_input_scaled[features.index('Tavg')] = pred_norm[0]
            new_input_scaled[features.index('RH_avg')] = pred_norm[1]
            new_input_scaled[features.index('RR')] = pred_norm[2]

            # Slide the input window and add the new data
            X_input = np.append(X_input[:, 1:, :], [[new_input_scaled]], axis=1)

    # --- 5. Display Weather Prediction Results ---
    df_pred = pd.DataFrame(list_prediksi, columns=['Temperature (°C)', 'Humidity (%)', 'Rainfall (mm)'])
    df_pred.index = np.arange(1, len(df_pred) + 1)
    df_pred.index.name = "Day"
    st.dataframe(df_pred.style.format("{:.2f}"), use_container_width=True)

    # ==============================================================================
    # CROP PREDICTION BASED ON WEATHER FORECAST
    # ==============================================================================
    st.header("🌾 Crop Recommendation")
    
    if not df_pred.empty:
        avg_temp = df_pred['Temperature (°C)'].mean()
        avg_humidity = df_pred['Humidity (%)'].mean()
        avg_rainfall = df_pred['Rainfall (mm)'].mean()

        crop_input = pd.DataFrame([[n, p, k, avg_temp, avg_humidity, ph, avg_rainfall]], 
                                  columns=['Nitrogen', 'Phosphorus', 'Potassium', 'Temperature', 'Humidity', 'pH_Value', 'Rainfall'])

        X_crop_scaled = crop_scaler.transform(crop_input)
        pred_crop = rf_crop_model.predict(X_crop_scaled)
        pred_label = label_encoder.inverse_transform(pred_crop)[0]

        st.success(f"Based on the soil conditions and the average weather forecast for the next {jumlah_hari_prediksi} days, the recommended crop is: **{pred_label.capitalize()}**")

        st.info(f"""
        **Summary of Predicted Conditions:**
        - **Average Temperature:** {avg_temp:.2f} °C
        - **Average Humidity:** {avg_humidity:.2f} %
        - **Average Rainfall:** {avg_rainfall:.2f} mm/day
        """)
    else:
        st.warning("Cannot provide crop recommendation because the weather forecast failed.")