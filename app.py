import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
import glob

# =====================================================
# 1️⃣ Titre et chargement des modèles
# =====================================================
st.title("📊 Prédiction des retraits GAB (Prophet + LSTM)")

# --- Charger le modèle LSTM ---
lstm_model = load_model("lstm_model.h5", compile=False)

# --- Charger les scalers ---
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# --- Charger les modèles Prophet par cluster ---
prophet_models = {}
for file in glob.glob("prophet_model_cluster_*.pkl"):
    cluster_id = int(file.split("_")[-1].split(".")[0])
    with open(file, "rb") as f:
        prophet_models[cluster_id] = pickle.load(f)

st.success("✅ Modèles chargés avec succès !")

# =====================================================
# 2️⃣ Entrée utilisateur pour prédiction individuelle
# =====================================================
num_gab = st.number_input("📌 Numéro du GAB", min_value=0, step=1)
date_arrete = st.date_input("📅 Date à prédire")

# --- Mapping num_gab → cluster (à adapter selon ton fichier CSV) ---
# Si tu as un CSV avec tous les GABs et clusters, tu peux le charger ici
# Exemple fictif :
gab_cluster_mapping = {
    101: 0,
    102: 1,
    103: 2,
    # ... compléter avec tous les GABs
}

if num_gab not in gab_cluster_mapping:
    st.warning("⚠️ Numéro de GAB inconnu !")
else:
    cluster_id = gab_cluster_mapping[num_gab]
    
    # --- Prévision Prophet ---
    model_prophet = prophet_models[cluster_id]
    future = model_prophet.make_future_dataframe(periods=1)
    forecast = model_prophet.predict(future)
    yhat = forecast[forecast['ds'] == pd.to_datetime(date_arrete)]['yhat'].values
    if len(yhat) == 0:
        st.warning("⚠️ Date en dehors de l'horizon de prédiction Prophet !")
    else:
        yhat = yhat[0]

        # --- Prévision LSTM sur les résidus ---
        # Ici, il faut les derniers résidus connus pour ce GAB
        # Exemple fictif : si tu n’as pas encore de CSV historique, on initialise à 0
        time_steps = 14
        last_resid = np.zeros(time_steps)  # remplacer par les vrais résidus si disponibles

        # Préparer l’entrée pour le LSTM
        X_input = scaler_X.transform(last_resid.reshape(1, -1)).reshape(1, time_steps, 1)
        y_resid_pred_scaled = lstm_model.predict(X_input)
        y_resid_pred = scaler_y.inverse_transform(y_resid_pred_scaled).flatten()[0]

        # --- Montant final prédit ---
        montant_pred_final = yhat + y_resid_pred

        st.subheader("💰 Prédiction finale pour ce GAB et cette date")
        st.write(f"Montant prévu : {montant_pred_final:.2f} DH")
