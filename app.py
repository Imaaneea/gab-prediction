import streamlit as st
import pandas as pd
import numpy as np
import pickle
import joblib
from tensorflow.keras.models import load_model
import glob
from sklearn.cluster import KMeans

st.title("📊 Prédiction des retraits GAB (Prophet + LSTM)")

# --- Charger le modèle LSTM et les scalers ---
lstm_model = load_model("lstm_model.h5", compile=False)
scaler_X = joblib.load("scaler_X.pkl")
scaler_y = joblib.load("scaler_y.pkl")

# --- Charger les modèles Prophet par cluster ---
prophet_models = {}
for file in glob.glob("prophet_model_cluster_*.pkl"):
    cluster_id = int(file.split("_")[-1].split(".")[0])
    with open(file, "rb") as f:
        prophet_models[cluster_id] = pickle.load(f)
st.success("✅ Modèles chargés avec succès !")

# --- Charger le CSV historique complet ---
df_hist = pd.read_csv("historique_gabs.csv", parse_dates=["date_arrete"])

# --- Clustering automatique des GABs ---
gab_features = df_hist.groupby("num_gab")["montant"].mean().reset_index()
k = len(prophet_models)
kmeans = KMeans(n_clusters=k, random_state=42)
gab_features["cluster"] = kmeans.fit_predict(gab_features[["montant"]])

# Mapping automatique num_gab -> cluster
gab_cluster_mapping = dict(zip(gab_features["num_gab"], gab_features["cluster"]))

# --- Entrée utilisateur ---
num_gab = st.number_input("📌 Numéro du GAB", min_value=0, step=1)

if num_gab not in df_hist["num_gab"].values:
    st.warning("⚠️ Numéro de GAB inconnu dans les données historiques !")
else:
    df_gab = df_hist[df_hist["num_gab"] == num_gab].sort_values("date_arrete")
    region = df_gab["region"].iloc[0]
    agence = df_gab["agence"].iloc[0]
    st.write(f"🏢 Région : {region}")
    st.write(f"🏦 Agence : {agence}")

    # --- Identifier le cluster automatiquement ---
    cluster_id = gab_cluster_mapping[num_gab]

    # --- Prédiction Prophet ---
    model_prophet = prophet_models[cluster_id]
    date_arrete = pd.Timestamp.today()
    future = model_prophet.make_future_dataframe(periods=1)
    forecast = model_prophet.predict(future)
    yhat = forecast[forecast['ds'] == date_arrete]['yhat'].values
    if len(yhat) == 0:
        st.warning("⚠️ Date en dehors de l'horizon de prédiction Prophet !")
    else:
        yhat = yhat[0]

        # --- Prévision LSTM sur les résidus ---
        time_steps = 14
        last_resid = df_gab["montant"].values[-time_steps:]
        if len(last_resid) < time_steps:
            last_resid = np.pad(last_resid, (time_steps - len(last_resid), 0))
        X_input = scaler_X.transform(last_resid.reshape(1, -1)).reshape(1, time_steps, 1)
        y_resid_pred_scaled = lstm_model.predict(X_input)
        y_resid_pred = scaler_y.inverse_transform(y_resid_pred_scaled).flatten()[0]

        montant_pred_final = yhat + y_resid_pred
        st.subheader("💰 Prédiction du jour pour ce GAB")
        st.write(f"Montant prévu : {montant_pred_final:.2f} DH")
