# File: analysis/user_analyzer.py (ML Upgraded Version with .pkl Persistence)
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import joblib
import os

MODEL_DIR = "models"
KMEANS_MODEL_PATH = os.path.join(MODEL_DIR, "user_persona_kmeans.pkl")
SCALER_PATH = os.path.join(MODEL_DIR, "user_persona_scaler.pkl")

def _prepare_user_features(df):
    """Helper function to aggregate session data to the user level."""
    user_df = df.groupby('user_id').agg(
        avg_duration_min=('duration_min', 'mean'),
        total_events=('events_count', 'sum'),
        session_count=('session_id', 'count'),
        device_count=('device_id', 'nunique'),
        avg_events_per_sec=('events_per_sec', 'mean')
    ).fillna(0)
    return user_df

def train_user_persona_model(df):
    """Trains a KMeans clustering model and saves it and its scaler to .pkl files."""
    print("\n[INFO] Starting model training for user persona clustering...")
    user_df = _prepare_user_features(df)
    
    features_to_cluster = ['avg_duration_min', 'total_events', 'session_count', 'device_count', 'avg_events_per_sec']
    
    scaler = StandardScaler()
    user_features_scaled = scaler.fit_transform(user_df[features_to_cluster])
    
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    kmeans.fit(user_features_scaled)
    print("[INFO] User persona model training complete.")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(kmeans, KMEANS_MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"[SUCCESS] KMeans model saved to '{KMEANS_MODEL_PATH}'")
    print(f"[SUCCESS] Scaler saved to '{SCALER_PATH}'")

def create_user_personas_with_saved_model(df):
    """Loads a pre-trained KMeans model to assign user personas."""
    print("\n[INFO] Loading pre-trained .pkl model for user persona clustering...")

    try:
        kmeans = joblib.load(KMEANS_MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError:
        print(f"[ERROR] User persona model not found at '{KMEANS_MODEL_PATH}'.")
        print("Please run the script in 'train' mode first.")
        df['persona_cluster'] = 'Model Not Found'
        return df

    user_df = _prepare_user_features(df)
    features_to_cluster = ['avg_duration_min', 'total_events', 'session_count', 'device_count', 'avg_events_per_sec']
    
    # Use the loaded scaler
    user_features_scaled = scaler.transform(user_df[features_to_cluster])
    
    # Use the loaded model
    user_df['persona_cluster'] = kmeans.predict(user_features_scaled)
    
    df = df.merge(user_df[['persona_cluster']], on='user_id', how='left')
    df.loc[df['persona_cluster'].notna(), 'anomaly_flags'] += 'User Persona Cluster ' + df['persona_cluster'].astype(str) + '; '
    
    print("[INFO] User persona clustering complete using saved model.")
    return df
