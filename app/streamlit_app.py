import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# Path absolut berdasarkan lokasi file ini
base_dir = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource
def load_artifacts():
    # Bangun path ke folder models (satu level di atas app)
    models_dir = os.path.join(base_dir, '..', 'models')
    
    # Debug: tampilkan path untuk verifikasi (hanya pertama kali)
    # if 'debug_shown' not in st.session_state:
    #     st.write(f"Base dir: {base_dir}")
    #     st.write(f"Models dir: {models_dir}")
    #     st.write(f"Files in models dir: {os.listdir(models_dir) if os.path.exists(models_dir) else 'DIR NOT FOUND'}")
    #     st.session_state.debug_shown = True
    
    model = joblib.load(os.path.join(models_dir, 'kprototypes_best.pkl'))
    scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
    encoders = joblib.load(os.path.join(models_dir, 'encoders.pkl'))
    return model, scaler, encoders

model, scaler, encoders = load_artifacts()

# Definisi fitur
numeric_cols = ['jumlah_transaksi', 'total_belanja', 'rata_rata_transaksi', 
                'std_transaksi', 'rata_rata_pajak', 'rentang_transaksi']
categorical_cols = ['produk_favorit', 'provider_terbanyak', 'channel_terbanyak']

st.set_page_config(page_title="Segmentasi Pelanggan Gachaku", layout="centered")
st.title("🎮 Segmentasi Pelanggan Gachaku")
st.markdown("Masukkan profil pelanggan untuk mengetahui segmennya.")

with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        jumlah = st.number_input("Jumlah Transaksi", min_value=1, value=2)
        total = st.number_input("Total Belanja (Rp)", min_value=0, value=200000)
        rata = st.number_input("Rata-rata per transaksi (Rp)", min_value=0, value=100000)
    with col2:
        std = st.number_input("Std dev transaksi (Rp)", min_value=0, value=50000)
        pajak_rata = st.number_input("Rata-rata pajak (Rp)", min_value=0, value=5000)
        rentang = st.number_input("Rentang hari (hari sejak transaksi pertama)", min_value=0, value=30)
    
    produk = st.selectbox("Produk favorit", options=['Robaks Limited 🔥 (Login)', 'Robaks Instant (Login)', 'Fish It Gamepasses', 'unknown'])
    provider = st.selectbox("Provider favorit", options=['duitku', 'tripay', 'unknown'])
    channel = st.selectbox("Channel favorit", options=['DUITKU', 'TRIPAY', 'unknown'])
    
    submitted = st.form_submit_button("Prediksi Segmen")

if submitted:
    numeric_input = np.array([[jumlah, total, rata, std, pajak_rata, rentang]])
    numeric_scaled = scaler.transform(numeric_input)
    
    cat_encoded = []
    for col, le in zip(categorical_cols, [encoders[c] for c in categorical_cols]):
        val = locals()[col.split('_')[0]]
        if val in le.classes_:
            cat_encoded.append(le.transform([val])[0])
        else:
            cat_encoded.append(0)
    cat_encoded = np.array([cat_encoded])
    
    X_input = np.hstack((numeric_scaled, cat_encoded))
    categorical_indices = list(range(6, 9))
    cluster = model.predict(X_input, categorical=categorical_indices)[0]
    
    segmen_map = {
        0: "🐟 Casual Player",
        1: "💎 High Spender (Whale)",
        2: "📱 Digital Wallet User (DANA)",
        3: "⚡ Robaks Instant User",
        4: "🔥 Limited Edition Hunter"
    }
    segmen = segmen_map.get(cluster, f"Cluster {cluster}")
    
    st.success(f"**Segmen Pelanggan:** {segmen}")
    st.info(f"Cluster ID: {cluster}")
    
    if cluster == 0:
        st.markdown("📌 **Rekomendasi:** Tawarkan bundle ROBUX eksklusif, program loyalitas tier platinum.")
    elif cluster == 1:
        st.markdown("📌 **Rekomendasi:** Promosi gamepass Fish It, diskon untuk pembelian ulang.")
    elif cluster == 2:
        st.markdown("📌 **Rekomendasi:** Kampanye cashback via DANA, topup minimal 50k.")
    else:
        st.markdown("📌 **Rekomendasi:** Bundle promosi untuk naik ke tier lebih tinggi.")