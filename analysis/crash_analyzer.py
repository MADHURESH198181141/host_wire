# File: analysis/crash_analyzer.py (ML Upgraded with .pkl Model Persistence)
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import joblib # Using joblib is still preferred for scikit-learn models
import os
import sys

# --- CHANGE HERE: Use .pkl extension for the saved model files ---
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "crash_predictor_model.pkl") # Changed extension to .pkl
ENCODER_PATH = os.path.join(MODEL_DIR, "crash_predictor_encoder.pkl") # Changed extension to .pkl

def _prepare_features(df):
    """Helper function to create features from the raw dataframe."""
    df['events_per_min'] = (df['events_count'] / df['duration_min']).replace([float('inf'), -float('inf')], 0).fillna(0)
    df['screens_per_min'] = (df['screens_viewed'] / df['duration_min']).replace([float('inf'), -float('inf')], 0).fillna(0)
    return df

def train_crash_model(df):
    """
    Trains a RandomForest model on the provided data and saves it to a .pkl file.
    """
    print("\n[INFO] Starting model training for crash prediction...")

    df = _prepare_features(df)

    # --- Feature Engineering ---
    categorical_features = ['platform', 'app_version']
    encoder = OneHotEncoder(handle_unknown='ignore')

    numerical_features = ['duration_min', 'events_per_min', 'screens_per_min']

    encoded_features = encoder.fit_transform(df[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    features_df = pd.concat([df[numerical_features].reset_index(drop=True), encoded_df], axis=1).fillna(0)

    target = df['crash']

    if len(df[df['crash']==True]) < 2:
        print("[WARNING] Not enough crash data to train a meaningful model. Aborting training.")
        return

    # --- Model Training ---
    model = RandomForestClassifier(random_state=42, class_weight='balanced')
    model.fit(features_df, target)
    print("[INFO] Model training complete.")

    # --- Save the Model and Encoder ---
    if not os.path.exists(MODEL_DIR):
        os.makedirs(MODEL_DIR)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(encoder, ENCODER_PATH)
    print(f"[SUCCESS] Model saved to '{MODEL_PATH}'")
    print(f"[SUCCESS] Encoder saved to '{ENCODER_PATH}'")

def predict_crash_risk_with_saved_model(df):
    """
    Loads a pre-trained .pkl model and uses it to predict crash risk on new data.
    """
    print("\n[INFO] Loading pre-trained .pkl model to predict crash risk...")

    try:
        model = joblib.load(MODEL_PATH)
        encoder = joblib.load(ENCODER_PATH)
    except FileNotFoundError:
        print(f"[ERROR] Model not found at '{MODEL_PATH}'.")
        print("Please run the script in 'train' mode first to create the model.")
        df['crash_risk'] = 'Model Not Found'
        return df

    df = _prepare_features(df)

    # --- Feature Engineering with Loaded Encoder ---
    categorical_features = ['platform', 'app_version']
    numerical_features = ['duration_min', 'events_per_min', 'screens_per_min']

    encoded_features = encoder.transform(df[categorical_features]).toarray()
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(categorical_features))
    features_df = pd.concat([df[numerical_features].reset_index(drop=True), encoded_df], axis=1).fillna(0)

    # --- Prediction ---
    crash_probabilities = model.predict_proba(features_df)[:, 1]

    df['crash_risk_probability'] = crash_probabilities
    df['crash_risk'] = pd.cut(crash_probabilities, bins=[-1, 0.3, 0.7, 1.1], labels=['Low', 'Medium', 'High'])
    df.loc[df['crash_risk'] == 'High', 'anomaly_flags'] += 'High Crash Risk (Predicted); '

    print("[INFO] Crash risk prediction complete.")
    return df
