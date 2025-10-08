# File: main.py (Upgraded with All Train/Predict models)
import pandas as pd
import sys
import os

from analysis import crash_analyzer, bot_detector, user_analyzer, session_analyzer, data_quality
from reporting import report_generator
from generate_bot_chart import create_bot_visualization
from generate_crash_chart import create_crash_visualization
from generate_duration_chart import create_duration_visualization

def preprocess_data(df):
    """Performs initial data cleaning and feature engineering."""
    df['session_start'] = pd.to_datetime(df['session_start'])
    df['session_end'] = pd.to_datetime(df['session_end'])
    df['duration_min'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60
    df['duration_sec'] = (df['session_end'] - df['session_start']).dt.total_seconds()
    df['crash'] = df['crash'].astype(bool)
    df['anomaly_flags'] = ''
    # This is needed by multiple modules, so we calculate it early
    df['events_per_sec'] = (df['events_count'] / df['duration_sec']).replace([float('inf'), -float('inf')], 0).fillna(0)
    return df

def main(filepath, mode='predict'):
    """
    Main function to run the entire analysis pipeline.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: The file '{filepath}' was not found.")
        return

    df = preprocess_data(df)

    if mode == 'train':
        print("--- RUNNING IN TRAIN MODE ---")
        crash_analyzer.train_crash_model(df)
        bot_detector.train_bot_model(df)
        user_analyzer.train_user_persona_model(df)
        # Add training functions for other models here if they are created
        print("\n[SUCCESS] All models have been trained and saved to the 'models' directory.")
        return

    elif mode == 'predict':
        print("--- RUNNING IN PREDICT MODE ---")
        # Use saved models for prediction and clustering
        df = crash_analyzer.predict_crash_risk_with_saved_model(df)
        df = bot_detector.detect_bots_with_saved_model(df)
        df = user_analyzer.create_user_personas_with_saved_model(df)

        # Unsupervised models that don't need pre-training can still be run directly
        df = session_analyzer.analyze_session_anomalies_ml(df)
        df = data_quality.check_and_prepare_data_quality(df)

        report_generator.create_full_report(df)

        # Generate Additional Charts
        create_bot_visualization(filepath)
        create_crash_visualization(filepath)
        create_duration_visualization(filepath)

if __name__ == "__main__":
    # If no arguments are given, default to 'predict' mode.
    if len(sys.argv) == 1:
        print("[INFO] No mode specified. Defaulting to 'predict' mode.")
        run_mode = 'predict'
        data_file_path = "data/app_usage_sessions.csv"
    # If arguments are given, use them.
    else:
        run_mode = sys.argv[1]
        data_file_path = sys.argv[2] if len(sys.argv) > 2 else "data/app_usage_sessions.csv"

        # Still validate the mode if it was provided.
        if run_mode not in ['train', 'predict']:
            print(f"Error: Invalid mode '{run_mode}'. Choose 'train' or 'predict'.")
            print("Usage: python main.py <mode> [data_file_path]")
            sys.exit(1)

    # Run the main function with the determined mode and path.
    main(data_file_path, run_mode)
