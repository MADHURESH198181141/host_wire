# File: analysis/data_quality.py (ML-Ready + Visualization Version)
import pandas as pd
from sklearn.impute import SimpleImputer
import matplotlib.pyplot as plt
import seaborn as sns
import os

def check_and_prepare_data_quality(df):
    """
    Checks for data quality issues, flags them, prepares the data for ML models,
    and generates a visualization of data completeness.
    """
    print("\n[INFO] Checking data quality and preparing data for ML models...")

    # --- 1. Flagging (No changes from before) ---
    df.loc[df['device_id'].isna(), 'anomaly_flags'] += 'Missing Device ID; '
    df.loc[(df['screens_viewed'] > 0) & ((df['events_count'] / df['screens_viewed']) > 50), 'anomaly_flags'] += 'High Event/Screen Ratio; '
    df.loc[(df['events_count'] == 0) & (df['crash'] == False), 'anomaly_flags'] += 'Zero Activity; '

    # --- 2. Visualization ---
    _create_data_quality_chart(df)

    # --- 3. Imputation (Preparing for ML) ---
    df['device_id'] = df['device_id'].fillna('UNKNOWN_DEVICE')

    return df

def _create_data_quality_chart(df, output_dir="reports"):
    """Generates a bar chart showing the percentage of missing data per column."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Calculate the percentage of non-missing values for each column
    completeness = (df.notna().sum() / len(df)) * 100
    completeness = completeness.reset_index()
    completeness.columns = ['Column', 'Completeness (%)']

    # Only show columns that are not 100% complete to highlight problems
    incomplete_cols = completeness[completeness['Completeness (%)'] < 100]

    if incomplete_cols.empty:
        print("[INFO] No missing data found. Data quality chart will not be generated.")
        return

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(12, 6))

    # Create the bar plot
    barplot = sns.barplot(
        data=incomplete_cols,
        x='Completeness (%)',
        y='Column',
        hue='Column',
        palette='coolwarm',
        legend=False
    )

    # Add text labels to the bars
    for index, row in incomplete_cols.iterrows():
        barplot.text(row['Completeness (%)'] - 5, index, f"{row['Completeness (%)']:.1f}%",
                     color='white', ha="center", va='center', fontweight='bold')

    plt.title('Data Completeness Report', fontsize=16, fontweight='bold')
    plt.xlabel('Percentage of Data Present (%)')
    plt.ylabel('Data Column')
    plt.xlim(0, 100) # Ensure the x-axis goes from 0 to 100

    chart_path = os.path.join(output_dir, 'data_completeness_report.png')
    plt.savefig(chart_path)
    plt.close()

    print(f"[Chart Saved] Data completeness chart saved to: '{chart_path}'")
