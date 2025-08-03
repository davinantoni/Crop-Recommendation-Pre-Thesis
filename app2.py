import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from tensorflow.keras.models import load_model
from sklearn.ensemble import RandomForestClassifier

# --- Load semua model dan scaler ---
@st.cache_resource
def load_models_and_scalers():
    lstm_models = {}
    lstm_scalers = {}

    model_folder = "models"
    scaler_folder = "scaler_lstm"

    for file in os.listdir(model_folder):
        if file.startswith("lstm_attention_") or file.startswith("finetuned_"):
            province = file.replace("lstm_attention_", "").replace("finetuned_", "").replace(".h5", "")
            lstm_models[province] = load_model(os.path.join(model_folder, file), compile=False)

    for file in os.listdir(scaler_folder):
        if file.startswith("scaler_lstm_"):
            province = file.replace("scaler_lstm_", "").replace(".pkl", "").replace("_", " ")
            lstm_scalers[province] = joblib.load(os.path.join(scaler_folder, file))

    rf_model = joblib.load("rf_crop_model.pkl")
    crop_scaler = joblib.load("crop_scaler.pkl")

    return lstm_models, lstm_scalers, rf_model, crop_scaler

lstm_models, lstm_scalers, rf_model, crop_scaler = load_models_and_scalers()

# --- UI Input ---
st.title("🌾 Rekomendasi Tanaman Berdasarkan Prediksi Cuaca")

st.subheader("Masukkan kondisi lahan:")
N = st.number_input("Nitrogen (N)", value=100)
P = st.number_input("Phosphorus (P)", value=45)
K = st.number_input("Potassium (K)", value=60)
pH = st.number_input("pH Tanah", value=6.4)

province = st.selectbox("Provinsi", sorted(lstm_models.keys()))
days_ahead = st.slider("Prediksi untuk berapa hari ke depan?", 30, 60, 30)


# --- Load Data Cuaca ---
@st.cache_data
def load_climate_data():
    df = pd.read_csv("climate_data_with_province.csv")
    df['date'] = pd.to_datetime(df['date'], dayfirst=True)
    return df

# --- Prediksi Cuaca ---
# @st.cache_data
# def predict_weather(province, days_ahead):
#     model = load_model(f"models/lstm_attention_{province}.h5", compile=False)
#     scaler = joblib.load(f"scaler_lstm/scaler_lstm_{province}.pkl")
#     df = load_climate_data()

#     # Ambil data cuaca historis untuk provinsi
#     province_data = df[df['province_name'] == province].copy()
#     province_data.sort_values('date', inplace=True)
    
#     # Ambil fitur yang digunakan dalam pelatihan
#     features = list(scaler.feature_names_in_)
#     print(features)
#     print(province_data.columns)
#     province_data = province_data[features]  # ambil dan urutkan kolom sesuai fitur saat fit
#     data_scaled = scaler.transform(province_data)
#     # data_scaled = scaler.transform(province_data[scaler.feature_names_in_])
    
#     # Ambil 30 langkah terakhir (input untuk prediksi)
#     window_size = 30
#     sequence = data_scaled[-window_size:].copy()
#     sequence = np.expand_dims(sequence, axis=0)  # Shape: (1, 30, 6)

#     preds = []

#     for _ in range(days_ahead):
#         pred = model.predict(sequence, verbose=0)[0]  # Shape: (6,)

#         # Buat input baru: geser sequence ke kiri dan tambahkan prediksi baru
#         next_step = np.concatenate([sequence[0, 1:, :], np.zeros((1, 6))], axis=0)  # Shape: (30,6)
        
#         # Ambil nilai sebelumnya untuk fitur yang tidak diprediksi
#         last_values = sequence[0, -1, :]  # Last timestep
#         pred_full = np.array([
#             pred[0],  # Tavg (diprediksi)
#             pred[1],  # RH_avg
#             pred[2],  # RR
#             last_values[3],  # Tn
#             last_values[4],  # Tx
#             last_values[5],  # ss
#         ])

#         next_step[-1, :] = pred_full
#         sequence = np.expand_dims(next_step, axis=0)

#         preds.append(pred_full[:3])  # Simpan hanya Tavg, RH_avg, RR

#     # Ambil rata-rata untuk 3 fitur utama (selama n hari ke depan)
#     preds = np.array(preds)
#     temp_avg, humidity_avg, rainfall_avg = scaler.inverse_transform(
#         np.hstack([preds, np.tile(last_values[3:], (len(preds), 1))])
#     )[:, :3].mean(axis=0)

#     return round(temp_avg, 1), round(humidity_avg, 1), round(rainfall_avg * days_ahead, 1)  # Total curah hujan

@st.cache_data
def predict_weather(province, days_ahead):
    lookback = 30
    features = ['Tn', 'Tx', 'Tavg', 'RH_avg', 'RR', 'ss']
    selected_features = ['Tavg', 'RH_avg', 'RR']

    # Load scaler dan model
    model = load_model(f"models/lstm_attention_{province}.h5", compile=False)
    scaler = joblib.load(f"scaler_lstm/scaler_lstm_{province}.pkl")

    # Load data dan filter berdasarkan provinsi
    df = pd.read_csv("climate_data_with_province.csv", parse_dates=["date"])
    df = df[df["province_name"] == province].copy()
    df.sort_values("date", inplace=True)

    # Isi missing value jika ada
    df[features] = df[features].fillna(method='ffill')

    # Simpan nilai-nilai asli untuk fitur non-target
    last_known = df[features].iloc[-1].copy()

    # Scaling data
    scaled_data = scaler.transform(df[features])
    input_seq = scaled_data[-lookback:].reshape(1, lookback, len(features))

    predictions = []

    for day in range(days_ahead):
        pred = model.predict(input_seq, verbose=0)[0]  # shape (6,)
        print(f"🔮 Scaled Prediction Day {day+1}:", pred)

        predictions.append(pred)

        # Ubah pred menjadi bentuk (1, 1, 6)
        pred_reshaped = pred.reshape(1, 1, len(features))

        # Update input_seq dengan pred terbaru
        input_seq = np.concatenate([input_seq[:, 1:, :], pred_reshaped], axis=1)


    # Konversi list ke array untuk inverse transform
    predictions = np.array(predictions)  # shape (days, 6)
    print("📈 All Predictions (scaled):", predictions)

    # Inverse scaling
    predictions_inv = scaler.inverse_transform(predictions)  # shape (days, 6)
    print("📊 All Predictions (inversed):", predictions_inv)

    # Ambil kolom Tavg, RH_avg, RR
    idx_map = {col: i for i, col in enumerate(scaler.feature_names_in_)}
    tavg_idx = idx_map['Tavg']
    rh_idx = idx_map['RH_avg']
    rr_idx = idx_map['RR']

    tavg_pred = predictions_inv[:, tavg_idx]
    rh_pred = predictions_inv[:, rh_idx]
    rr_pred = predictions_inv[:, rr_idx]

    # Cek hasil
    print("🌡️ Tavg:", tavg_pred)
    print("💧 RH_avg:", rh_pred)
    print("🌧️ RR:", rr_pred)

    # Hitung rata-rata prediksi
    return (
        round(np.mean(tavg_pred), 2),
        round(np.mean(rh_pred), 2),
        round(np.mean(rr_pred), 2)
    )


if st.button("🔍 Prediksi dan Rekomendasikan Tanaman"):
    with st.spinner("Memproses prediksi cuaca dan rekomendasi..."):
        temp, humidity, rainfall = predict_weather(province, days_ahead)

        # Tampilkan hasil prediksi cuaca
        st.subheader("🌤️ Rata-rata Cuaca Prediksi:")
        st.write(f"- Temperatur: {temp:.2f} °C")
        st.write(f"- Kelembapan: {humidity:.2f} %")
        st.write(f"- Curah Hujan: {rainfall:.2f} mm")

        # Siapkan input untuk RF
        input_features = np.array([[N, P, K, temp, humidity, pH, rainfall]])
        input_scaled = crop_scaler.transform(input_features)
        crop_pred = rf_model.predict(input_scaled)[0]

        # Output rekomendasi
        st.success(f"🌱 Rekomendasi tanaman: **{crop_pred}**")
