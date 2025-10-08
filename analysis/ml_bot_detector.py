# File: ml_bot_detector.py
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import sys
import os

def detect_bots_with_ml(df):
    """
    Uses the Isolation Forest unsupervised learning model to detect anomalous sessions.

    Args:
        df (pd.DataFrame): The preprocessed DataFrame with session data.

    Returns:
        pd.DataFrame: The original DataFrame with an added 'bot_risk_ml' column.
    """
    print("--- Running Bot Detection with Isolation Forest ML Model ---")

    # 1. Select features for the model to analyze.
    # These features numerically describe the behavior of a session.
    features = ['duration_sec', 'events_count', 'screens_viewed', 'events_per_sec', 'screens_per_sec']
    
    # 2. Prepare the data for the model.
    # The model requires a clean, numerical dataset with no missing values.
    print(f"\n[INFO] Using the following features for anomaly detection: {features}")
    df_features = df[features].fillna(0)
    
    # Handle potential infinite values that can result from division by zero
    df_features.replace([np.inf, -np.inf], 0, inplace=True)

    # 3. Initialize and train the Isolation Forest model.
    # - `contamination='auto'` is a modern default that often works well. It helps the
    #   model decide on its own how many anomalies to expect.
    # - `random_state=42` ensures that the model produces the same results every time
    #   it's run, which is good for reproducibility.
    model = IsolationForest(contamination='auto', random_state=42, n_estimators=100)
    
    print("[INFO] Training the model to learn what 'normal' session behavior looks like...")
    predictions = model.fit_predict(df_features)

    # 4. Interpret the model's predictions.
    # The model outputs -1 for anomalies (which we label as potential bots) and 1 for
    # inliers (which we label as normal sessions).
    print("[INFO] Predicting which sessions are anomalies (potential bots)...")
    df['bot_risk'] = ['High' if x == -1 else 'Low' for x in predictions]
    
    return df

def preprocess_data(filepath=None):
    """Loads and preprocesses the session data from a CSV file."""
    if filepath is None:
        filepath = os.path.join(os.path.dirname(__file__), "../data/app_usage_sessions.csv")
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'")
        sys.exit(1)
        
    df['session_start'] = pd.to_datetime(df['session_start'])
    df['session_end'] = pd.to_datetime(df['session_end'])
    df['duration_sec'] = (df['session_end'] - df['session_start']).dt.total_seconds()
    
    # Calculate rates per second
    df['events_per_sec'] = df['events_count'] / df['duration_sec']
    df['screens_per_sec'] = df['screens_viewed'] / df['duration_sec']
    
    return df

# --- Main execution block ---
if __name__ == "__main__":
    # You can optionally pass a different file path as a command-line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else None

    # Load and prepare the data
    session_df = preprocess_data(file_path)
    
    # Run the ML-based bot detection
    df_with_predictions = detect_bots_with_ml(session_df)
    
    # --- Generate the final report ---
    print("\n" + "="*60)
    print("      Machine Learning Anomaly Detection Report")
    print("="*60)

    # Filter for only the sessions flagged as high risk by the model
    high_risk_sessions = df_with_predictions[df_with_predictions['bot_risk'] == 'High']

    if high_risk_sessions.empty:
        print("\n[SUCCESS] The ML model did not find any significant anomalies in the dataset.")
    else:
        print(f"\n[RESULT] The model identified {len(high_risk_sessions)} sessions as 'High Risk' anomalies:")
        
        # Display the details of the flagged sessions
        report_columns = [
            'session_id',
            'user_id',
            'duration_sec',
            'events_count',
            'screens_viewed',
            'bot_risk'
        ]
        print(high_risk_sessions[report_columns].round(2).to_markdown(index=False))
