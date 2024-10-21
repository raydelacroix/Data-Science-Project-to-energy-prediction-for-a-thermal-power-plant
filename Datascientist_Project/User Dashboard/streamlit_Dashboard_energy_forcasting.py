import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import tensorflow as tf
from tensorflow.keras.models import load_model
import plotly.express as px
from datetime import datetime, timedelta
import json
import seaborn as sns
import matplotlib.pyplot as plt
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import influxdb_client
from influxdb_client.client.query_api import QueryApi
import time

# Configuration InfluxDB
INFLUXDB_URL = "http://localhost:8086"
INFLUXDB_TOKEN = "L1kzKyVVGvfnJdV1RDoyDXHZe1OXFuOhbShcz2SlqJ7J4ehWCR6jJtV1uul-vwiZtNPYm_XzZ0eI-0tAmoON1w=="
INFLUXDB_ORG = "Projet"
INFLUXDB_BUCKET = "DATA_STREAMING_OPC_UA"

# Configuration Email et Pushbullet
SENDER_EMAIL = '***********'
SENDER_PASSWORD = '**********'
SMTP_SERVER = 'smtp.gmail.com'
SMTP_PORT = 587

def send_pushbullet_notification(token, title, body):
    url = "https://api.pushbullet.com/v2/pushes"
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json"
    }
    data = {
        "type": "note",
        "title": title,
        "body": body
    }

    try:
        response = requests.post(url, json=data, headers=headers)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Erreur lors de l'envoi de la notification Pushbullet : {e}")
        raise

def send_email(sender_email, sender_password, recipient_email, subject, message):
    msg = MIMEMultipart('alternative')
    msg['From'] = sender_email
    msg['To'] = recipient_email
    msg['Subject'] = subject

    html = f'''
    <html>
        <body style="font-family: Arial, sans-serif; padding: 20px;">
            <h2 style="color: #d9534f;">Alertes de dépassement de seuils</h2>
            <div style="background-color: #f8f9fa; padding: 15px; border-radius: 5px;">
                <pre style="white-space: pre-wrap;">{message}</pre>
            </div>
            <p style="color: #666; margin-top: 20px; font-size: 12px;">
                Ce message a été généré automatiquement par le système de surveillance.
            </p>
        </body>
    </html>
    '''

    text_part = MIMEText(message, 'plain')
    html_part = MIMEText(html, 'html')
    msg.attach(text_part)
    msg.attach(html_part)

    try:
        with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.sendmail(sender_email, recipient_email, msg.as_string())
            print("Email envoyé avec succès!")
    except Exception as e:
        print(f"Erreur lors de l'envoi de l'email : {str(e)}")
        raise

def check_thresholds_and_send_alerts(client, token, recipient_email):
    query_api = client.query_api()
    query = f'''
    from(bucket: "{INFLUXDB_BUCKET}")
    |> range(start: -2m)
    |> filter(fn: (r) => r["_measurement"] == "H_Power" or 
                       r["_measurement"] == "Fact_puiss_Rezo_Elct" or 
                       r["_measurement"] == "Rotation_Turbine__C'" or 
                       r["_measurement"] == "Frequence_Electricity_f_RezoElect" or 
                       r["_measurement"] == "QSS_B" or 
                       r["_measurement"] == "QFW_A" or 
                       r["_measurement"] == "ΔPSH1_C" or 
                       r["_measurement"] == "Tension_elect_RezoElectr")
    |> filter(fn: (r) => r["_field"] == "value")
    |> aggregateWindow(every: 2m, fn: mean, createEmpty: false)
    |> yield(name: "mean")
    '''

    try:
        tables = query_api.query(query=query)
    except Exception as e:
        print(f"Erreur lors de l'exécution de la requête : {e}")
        return

    thresholds = {
        "H_Power": 10,
        "Fact_puiss_Rezo_Elct": 10,
        "Rotation_Turbine__C'": 10,
        "Frequence_Electricity_f_RezoElect": 10,
        "QSS_B": 10,
        "QFW_A": 10,
        "ΔPSH1_C": 10,
        "Tension_elect_RezoElectr": 10
    }

    alerts = []
    
    for table in tables:
        for record in table.records:
            try:
                measurement = record.values["_measurement"]
                value = record.values["_value"]
                
                if measurement in thresholds and value > thresholds[measurement]:
                    alerts.append(f"- {measurement}: {value:.2f} (seuil: {thresholds[measurement]})")
            except KeyError as e:
                print(f"Clé manquante dans le record : {e}")

    if alerts:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        message = f"Alertes de dépassement de seuils ({timestamp}):\n\n" + "\n".join(alerts)
        
        try:
            send_pushbullet_notification(token, "Alertes de dépassement de seuils", message)
            print("Notification Pushbullet envoyée avec succès")
        except Exception as e:
            print(f"Erreur lors de l'envoi de la notification Pushbullet : {e}")

        try:
            send_email(SENDER_EMAIL, SENDER_PASSWORD, recipient_email, 
                      "Alertes de dépassement de seuils", message)
            print("Email envoyé avec succès")
        except Exception as e:
            print(f"Erreur lors de l'envoi de l'email : {e}")

        return True
    
    return False

def load_xgboost_model(file_path):
    encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
    for encoding in encodings:
        try:
            with open(file_path, 'r', encoding=encoding) as f:
                model_json = json.load(f)
            model = xgb.Booster()
            model.load_model(file_path)
            return model
        except UnicodeDecodeError:
            continue
        except Exception as e:
            st.error(f"Erreur lors du chargement du modèle avec l'encodage {encoding}: {str(e)}")
    
    raise ValueError("Impossible de charger le modèle XGBoost avec les encodages disponibles")

def create_prediction_dataframe(start_date_str, period='1w'):  
    start_date = pd.Timestamp(start_date_str)  
    
    if period == '1d':  
        periods = 24 * 6  # 144 points pour un jour  
    elif period == '1w':  
        periods = 7 * 24 * 6  # 1008 points pour une semaine  
    else:  
        raise ValueError("Période non valide. Utilisez '1d' ou '1w'.")  
    
    predict_period_dates = pd.date_range(start=start_date, periods=periods, freq='10min')  
    
    real_data = np.random.uniform(low=30, high=40, size=len(predict_period_dates))  
    predicted_data = np.random.uniform(low=30, high=40, size=len(predict_period_dates))  
    error_data = real_data - predicted_data
    
    df = pd.DataFrame({  
        'date': predict_period_dates,  
        'real': real_data,  
        'predicted': predicted_data,
        'erreur': error_data  
    })  
    
    return df

def main():
    st.title('Application d\'Analyse et Prédiction de H_Power')

    menu = st.sidebar.selectbox(
        'Menu',
        ['Prédiction de H_Power', 'Configuration de notification']
    )

    if menu == 'Prédiction de H_Power':
        st.header('Prédiction de H_Power')

        model_option = st.selectbox(
            'Choisissez le modèle',
            ['XGBoost', 'LSTM']
        )

        if model_option == 'XGBoost':
            model_path = 'chemin/XGBOOST_h_power_model_predicted.json'
            try:
                model = load_xgboost_model(model_path)
                st.success("Modèle XGBoost chargé avec succès")
            except Exception as e:
                st.error(f"Une erreur est survenue lors du chargement du modèle XGBoost : {str(e)}")
                st.stop()
        else:
            model_path = 'chemin/LSTM_h_power_model_predicted.h5'
            try:
                model = load_model(model_path)
                st.success("Modèle LSTM chargé avec succès")
            except Exception as e:
                st.error(f"Une erreur est survenue lors du chargement du modèle LSTM : {str(e)}")
                st.stop()

        start_date = st.text_input('Entrez la date de début (format YYYY-MM-DD HH:MM:SS)', '2023-11-01 00:00:00')  
        period = st.selectbox('Choisissez la période de prédiction', ['1d', '1w'])  

        if st.button('Générer les Prédictions'):  
            df = create_prediction_dataframe(start_date, period)  
            
            st.write('### Données Prédites')  
            
            today = datetime.now()  
            chosen_date = pd.to_datetime(start_date)  

            if chosen_date >= today + timedelta(days=1):  
                df_display = df[['date', 'predicted']]  
                st.write(df_display)  
                st.write('### Résumé des Données')  
                st.write(df_display.describe())  
                
                fig = px.line(df_display, x='date', y='predicted', title='Graphique des Prédictions', labels={'predicted': 'Puissance Prédite'})  
                st.plotly_chart(fig)  
            else:  
                df_display = df[['date', 'real', 'predicted', 'erreur']]  
                st.write(df_display)  
                st.write(df_display.describe()) 
                
                fig = px.line(df_display, x='date', y=['real', 'predicted'], title='Graphique des Prédictions', labels={'value': 'Puissance', 'variable': 'Type'})  
                st.plotly_chart(fig)

    elif menu == 'Configuration de notification':
        st.header("Configuration de notification")
        col1, col2 = st.columns(2)
        
        with col1:
            recipient_email = st.text_input("Adresse e-mail du destinataire")
        with col2:
            access_token = st.text_input("Token d'accès Pushbullet")

        check_frequency = st.slider("Fréquence de vérification (minutes)", min_value=1, max_value=60, value=2)
        num_checks = st.slider("Nombre de vérifications", min_value=1, max_value=24, value=5)

        if st.button("Démarrer la surveillance", type="primary"):
            if recipient_email and access_token:
                client = influxdb_client.InfluxDBClient(
                    url=INFLUXDB_URL, 
                    token=INFLUXDB_TOKEN, 
                    org=INFLUXDB_ORG
                )
                
                progress_bar = st.progress(0)
                status_text = st.empty()

                for i in range(num_checks):
                    status_text.text(f"Vérification {i+1}/{num_checks} en cours...")
                    
                    try:
                        with st.spinner('Surveillance en cours...'):
                            alert_sent = check_thresholds_and_send_alerts(
                                client, 
                                access_token, 
                                recipient_email
                            )
                            
                            if alert_sent:
                                st.success(f"✅ Alertes envoyées avec succès! (Vérification {i+1})")
                            else:
                                st.info(f"ℹ️ Aucun dépassement de seuil détecté (Vérification {i+1})")
                    
                    except Exception as e:
                        st.error(f"❌ Erreur lors de la vérification {i+1}: {str(e)}")
                    
                    progress_bar.progress((i + 1) / num_checks)
                    
                    if i < num_checks - 1:
                        time.sleep(check_frequency * 60)
                
                status_text.text("Surveillance terminée!")
                st.success("✅ Programme de surveillance terminé")
            else:
                st.error("⚠️ Veuillez remplir tous les champs.")

if __name__ == "__main__":
    main()
