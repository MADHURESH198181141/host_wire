# File: generate_crash_chart.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys

def create_crash_visualization(filepath="data/app_usage_sessions.csv"):
    """
    Loads session data and generates a bar chart of crash counts by app version and platform.
    """
    try:
        df = pd.read_csv(filepath)
    except FileNotFoundError:
        print(f"Error: Data file not found at '{filepath}'")
        print("Please make sure 'app_usage_sessions.csv' is in the same directory.")
        return

    # Filter for only the sessions that crashed
    crashed_sessions = df[df['crash'] == True]

    if crashed_sessions.empty:
        print("No crashed sessions found in the data. No chart will be generated.")
        return

    # --- Chart Generation ---
    output_dir = "reports"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(10, 7))

    # Create the bar plot using seaborn's countplot
    sns.countplot(
        data=crashed_sessions,
        x='app_version',
        hue='platform',
        palette={'iOS': '#007AFF', 'Android': '#34C759'}, # Apple and Android brand colors
        order=crashed_sessions['app_version'].value_counts().index # Order by most frequent
    )

    plt.title('Crash Counts by App Version and Platform', fontsize=18, fontweight='bold')
    plt.xlabel('App Version', fontsize=12)
    plt.ylabel('Number of Crashes', fontsize=12)
    plt.xticks(rotation=0)
    plt.legend(title='Platform')
    plt.tight_layout() # Adjust plot to ensure everything fits without overlapping

    # --- Save the Chart ---
    chart_path = os.path.join(output_dir, 'crash_summary_chart.png')
    plt.savefig(chart_path)

    print(f"✅ Success! Crash analysis chart has been saved to: '{chart_path}'")

if __name__ == "__main__":
    # You can optionally pass a different file path as a command-line argument
    file_path = sys.argv[1] if len(sys.argv) > 1 else "data/app_usage_sessions.csv"
    create_crash_visualization(file_path)
