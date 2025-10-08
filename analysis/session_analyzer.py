# File: analysis/session_analyzer.py (ML + Visualization Version)
import pandas as pd
import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_session_anomalies_ml(df):
    """
    Uses the Local Outlier Factor (LOF) model to detect anomalous session behaviors
    and generates a visualization of the findings.
    """
    # --- 1. Feature Engineering ---
    df['events_per_min'] = (df['events_count'] / df['duration_min']).replace([np.inf, -np.inf], 0).fillna(0)

    features = ['duration_min', 'events_count', 'screens_viewed', 'events_per_min']
    df_features = df[features].fillna(0)

    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df_features)

    # --- 2. Anomaly Detection with LOF ---
    lof = LocalOutlierFactor(n_neighbors=5, contamination='auto')
    predictions = lof.fit_predict(features_scaled)
    df['is_session_anomaly'] = (predictions == -1)

    # --- 3. Flagging ---
    df.loc[df['is_session_anomaly'], 'anomaly_flags'] += 'Anomalous Session (ML); '
    print("\n[INFO] Session behavior anomalies identified using Local Outlier Factor.")

    # --- 4. Visualization ---
    _create_session_anomaly_chart(df)

    return df

def _create_session_anomaly_chart(df, output_dir="reports"):
    """Generates a 2D density plot to visualize session activity hotspots."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.figure(figsize=(11, 7))

    # Create a joint plot which combines a scatter plot with density contours
    g = sns.jointplot(
        data=df,
        x='duration_min',
        y='events_count',
        hue='is_session_anomaly',
        palette={True: '#FF3B30', False: '#007AFF'}, # Red for anomalies, Blue for normal
        alpha=0.8
    )

    # Add a density estimation layer (kde) to show the "hotspots"
    g.plot_joint(sns.kdeplot, color="black", zorder=0, levels=6, alpha=0.5)

    g.fig.suptitle("Hotspots of User Activity: Session Duration vs. Event Count", y=1.02, fontsize=16, fontweight='bold')
    g.set_axis_labels('Session Duration (minutes)', 'Event Count', fontsize=12)

    # Adjust legend
    handles, labels = g.ax_joint.get_legend_handles_labels()
    g.ax_joint.legend(handles=handles, labels=['Normal Session', 'ML Anomaly'], title='Session Type')

    chart_path = os.path.join(output_dir, 'session_activity_hotspots.png')
    g.savefig(chart_path)
    plt.close('all') # Close all figures to free up memory

    print(f"[Chart Saved] Session activity hotspot chart saved to: '{chart_path}'")
