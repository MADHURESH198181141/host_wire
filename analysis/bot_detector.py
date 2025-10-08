# File: analysis/bot_detector.py (ML Model Version with .pkl Persistence)
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
import os

MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "bot_detector_model.pkl")

def _prepare_features(df):
    """Helper function to create features for the bot detection model."""
    df['events_per_sec'] = (df['events_count'] / df['duration_sec']).replace([np.inf, -np.inf], 0).fillna(0)
    df['screens_per_sec'] = (df['screens_viewed'] / df['duration_sec']).replace([np.inf, -np.inf], 0).fillna(0)
    return df

def train_bot_model(df):
    """Trains an IsolationForest model and saves it to a .pkl file."""
    print("\n[INFO] Starting model training for bot detection...")
    df = _prepare_features(df)
    
    features = ['duration_sec', 'events_count', 'screens_viewed', 'events_per_sec', 'screens_per_sec']
    df_features = df[features].fillna(0)

    model = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
    model.fit(df_features)
    print("[INFO] Bot detection model training complete.")
    
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)
    joblib.dump(model, MODEL_PATH)
    print(f"[SUCCESS] Bot detection model saved to '{MODEL_PATH}'")

def detect_bots_with_saved_model(df):
    """Loads a pre-trained IsolationForest model to detect anomalies."""
    print("\n[INFO] Loading pre-trained .pkl model for bot detection...")
    
    try:
        model = joblib.load(MODEL_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Bot model not found at '{MODEL_PATH}'.")
        print("Please run the script in 'train' mode first.")
        df['bot_risk'] = 'Model Not Found'
        return df
        
    df = _prepare_features(df)
    features = ['duration_sec', 'events_count', 'screens_viewed', 'events_per_sec', 'screens_per_sec']
    df_features = df[features].fillna(0)
    
    predictions = model.predict(df_features)
    
    df['bot_risk'] = np.where(predictions == -1, 'High', 'Low')
    df['bot_score'] = np.where(df['bot_risk'] == 'High', 5, 0)
    
    high_risk_users = df[df['bot_risk'] == 'High']['user_id'].unique()
    user_session_counts = df[df['user_id'].isin(high_risk_users)]['user_id'].value_counts()
    repeat_offenders = user_session_counts[user_session_counts > 1].index
    df.loc[df['user_id'].isin(repeat_offenders) & (df['bot_risk'] == 'High'), 'anomaly_flags'] += 'ML Detected Repeat Offender; '
    
    print("[INFO] Bot detection complete using saved model.")
    return df
