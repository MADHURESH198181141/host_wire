# File: generate_duration_chart.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def create_duration_visualization(filepath="data/app_usage_sessions.csv"):
    """
    Loads session data and visualizes the distribution of session durations.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'")
        print("Please make sure 'app_usage_sessions.csv' is in the same directory.")
        return

    # --- Preprocessing ---
    df['session_start'] = pd.to_datetime(df['session_start'])
    df['session_end'] = pd.to_datetime(df['session_end'])
    df['duration_min'] = (df['session_end'] - df['session_start']).dt.total_seconds() / 60

    # Filter out sessions with no duration to avoid errors
    valid_durations = df['duration_min'].dropna()

    # --- Chart Generation ---
    output_dir = "reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 6))

    # Create a histogram with a Kernel Density Estimate (KDE) line
    sns.histplot(
        data=valid_durations,
        bins=20,
        kde=True,
        color='#5856D6' # A nice purple
    )

    plt.title('Distribution of User Session Durations', fontsize=18, fontweight='bold')
    plt.xlabel('Session Duration (minutes)', fontsize=12)
    plt.ylabel('Number of Sessions (Frequency)', fontsize=12)
    plt.tight_layout()

    # --- Save the Chart ---
    chart_path = os.path.join(output_dir, 'session_duration_distribution.png')
    plt.savefig(chart_path)

    print(f"✅ Success! Session duration chart has been saved to: '{chart_path}'")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/app_usage_sessions.csv"
    create_duration_visualization(file_path)
