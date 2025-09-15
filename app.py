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

# --- Charger le CSV historique contenant les GABs et leurs montants ---
uploaded_file = st.file_uploader("📂 Importer le fichier historique GAB (CSV)", type=["csv"])

if uploaded_file:
    df_hist = pd.read_csv(uploaded_file, parse_dates=["date_arrete"])
    
    # =====================================================
    # Clustering automatique des GABs selon les données historiques
    # =====================================================
    gab_features = df_hist.groupby("num_gab")["montant"].mean().reset_index()
    k = len(prophet_models)  # nombre de clusters = nombre de modèles Prophet
    kmeans = KMeans(n_clusters=k, random_state=42)
    gab_features["cluster"] = kmeans.fit_predict(gab_features[["montant"]])
    
    # Mapping automatique num_gab -> cluster
    gab_cluster_mapping = dict(zip(gab_features["num_gab"], gab_features["cluster"]))
    
    # =====================================================
    # Entrée utilisateur pour prédiction
    # =====================================================
    num_gab = st.number_input("📌 Numéro du GAB", min_value=0, step=1)
    date_arrete = st.date_input("📅 Date à prédire")
    
    if num_gab not in gab_cluster_mapping:
        st.warning("⚠️ Numéro de GAB inconnu dans les données historiques !")
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
            # Récupérer les derniers résidus connus pour ce GAB
            df_resid = df_hist.copy()
            df_resid['residu'] = 0  # Placeholder si tu n’as pas les vrais résidus
            df_gab = df_resid[df_resid["num_gab"] == num_gab].sort_values("date_arrete")
            time_steps = 14
            last_resid = df_gab["residu"].values[-time_steps:]
            if len(last_resid) < time_steps:
                last_resid = np.pad(last_resid, (time_steps - len(last_resid), 0))
            
            X_input = scaler_X.transform(last_resid.reshape(1, -1)).reshape(1, time_steps, 1)
            y_resid_pred_scaled = lstm_model.predict(X_input)
            y_resid_pred = scaler_y.inverse_transform(y_resid_pred_scaled).flatten()[0]
            
            montant_pred_final = yhat + y_resid_pred
            
            st.subheader("💰 Prédiction finale pour ce GAB et cette date")
            st.write(f"Montant prévu : {montant_pred_final:.2f} DH")
