# File: generate_bot_chart.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import numpy as np

def create_bot_visualization(filepath="data/app_usage_sessions.csv"):
    """
    Loads session data and generates a scatter plot to identify bot-like outliers.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'")
        print("Please make sure 'app_usage_sessions.csv' is in the same directory.")
        return

    # --- Preprocessing and Bot Detection Logic ---
    df['session_start'] = pd.to_datetime(df['session_start'])
    df['session_end'] = pd.to_datetime(df['session_end'])
    df['duration_sec'] = (df['session_end'] - df['session_start']).dt.total_seconds()
    df['events_per_sec'] = (df['events_count'] / df['duration_sec']).replace(np.inf, 0)

    # Flag sessions as 'bot' based on extreme event rates
    df['is_bot_flagged'] = df['events_per_sec'] > 100

    # --- Chart Generation ---
    output_dir = "reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(11, 7))

    # Create the scatter plot
    sns.scatterplot(
        data=df,
        x='duration_sec',
        y='events_count',
        hue='is_bot_flagged',
        palette={True: '#FF3B30', False: '#007AFF'}, # Red for bots, Blue for normal
        alpha=0.7,
        s=50 # Marker size
    )

    plt.title('Bot-like Activity vs. Normal Sessions', fontsize=18, fontweight='bold')
    plt.xlabel('Session Duration (seconds) - Log Scale', fontsize=12)
    plt.ylabel('Event Count - Log Scale', fontsize=12)

    # Use a logarithmic scale to better visualize extreme outliers
    plt.xscale('log')
    plt.yscale('log')

    plt.legend(title='Bot Detected?', labels=['Yes', 'No'])
    plt.tight_layout()

    # --- Save the Chart ---
    chart_path = os.path.join(output_dir, 'bot_activity_chart.png')
    plt.savefig(chart_path)

    print(f"✅ Success! Bot activity chart has been saved to: '{chart_path}'")

if __name__ == "__main__":
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/app_usage_sessions.csv"
    create_bot_visualization(file_path)
