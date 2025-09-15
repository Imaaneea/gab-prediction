import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt

# =====================================================
# 1. Charger les modèles et scalers
# =====================================================
st.title("📊 Prédiction des retraits GAB (Prophet + LSTM)")

# Charger LSTM
lstm_model = load_model("lstm_model.keras")  # ⚠️ utilise .keras si tu l’as sauvegardé comme ça

# Charger scalers
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# Charger Prophet par cluster
prophet_models = {}
import glob
for file in glob.glob("prophet_model_cluster_*.pkl"):
    cluster_id = file.split("_")[-1].split(".")[0]
    with open(file, "rb") as f:
        prophet_models[cluster_id] = pickle.load(f)

st.success("✅ Modèles chargés avec succès !")

# =====================================================
# 2. Importer un fichier de données utilisateur
# =====================================================
uploaded_file = st.file_uploader("📂 Importer vos données GAB (CSV)", type=["csv"])

if uploaded_file:
    df_new = pd.read_csv(uploaded_file, parse_dates=["date_arrete"])
    st.write("Aperçu des données importées :")
    st.dataframe(df_new.head())

    # =====================================================
    # 3. Choisir un cluster
    # =====================================================
    clusters = df_new["cluster"].unique()
    selected_cluster = st.selectbox("Sélectionnez un cluster :", clusters)

    # =====================================================
    # 4. Prédiction avec Prophet
    # =====================================================
    model_prophet = prophet_models[str(selected_cluster)]
    future = model_prophet.make_future_dataframe(periods=7)  # ex : 7 jours à prévoir
    forecast = model_prophet.predict(future)

    st.subheader("📈 Prévisions Prophet")
    st.line_chart(forecast.set_index("ds")[["yhat"]])

    # =====================================================
    # 5. Prévision avec LSTM (sur résidus)
    # =====================================================
    st.subheader("🤖 Ajustement LSTM sur les résidus")
    df_cluster = df_new[df_new["cluster"] == selected_cluster].sort_values("date_arrete")
    values = df_cluster["montant"].values

    time_steps = 14
    if len(values) > time_steps:
        X = []
        for i in range(len(values) - time_steps):
            X.append(values[i:i+time_steps])
        X = np.array(X)

        X_scaled = scaler_X.transform(X.reshape(-1, time_steps)).reshape(X.shape)
        X_lstm = X_scaled.reshape((X_scaled.shape[0], X_scaled.shape[1], 1))

        y_pred_scaled = lstm_model.predict(X_lstm)
        y_pred = scaler_y.inverse_transform(y_pred_scaled)

        st.line_chart(y_pred[-100:])  # afficher dernières prédictions
