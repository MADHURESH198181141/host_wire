import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def create_full_report(df, output_dir="reports"):
    """Generates all text and visual reports."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    pd.set_option('display.max_rows', None)
    
    print("\n" + "="*80)
    print("App Session Analysis Full Report")
    print("="*80)

    _print_session_by_session_report(df)
    _print_crash_summary(df)
    _print_user_device_report(df)
    _print_bot_summary(df)
    _print_recommendations()
    _generate_visuals(df, output_dir)

def _print_session_by_session_report(df):
    print("\n\n--- 1. Session-by-Session Analysis ---")
    report_df = df[[
        'session_id', 'user_id', 'duration_min', 'events_count', 'screens_viewed', 'crash', 'bot_risk', 'anomaly_flags'
    ]].copy()
    report_df['duration_min'] = report_df['duration_min'].round(1)
    report_df['anomaly_flags'] = report_df['anomaly_flags'].str.strip().str.rstrip(';')
    print(report_df.to_markdown(index=False))

def _print_crash_summary(df):
    print("\n\n--- 2. Crash Analysis Summary ---")
    crashed_sessions = df[df['crash'] == True]
    summary = crashed_sessions.groupby(['app_version', 'platform']).size().reset_index(name='crash_count')
    print(summary.to_markdown(index=False))

def _print_user_device_report(df):
    print("\n\n--- 3. User-Level Multiple Device Report ---")
    user_device_map = df.dropna(subset=['device_id']).groupby('user_id')['device_id'].unique().apply(list).reset_index()
    multi_device_users = user_device_map[user_device_map['device_id'].apply(len) > 1]
    if not multi_device_users.empty:
        print(multi_device_users.to_markdown(index=False))
    else:
        print("No users with multiple devices found.")

def _print_bot_summary(df):
    print("\n\n--- 4. Bot Behavior Detection Summary ---")
    bot_df = df[df['bot_score'] >= 2][['session_id', 'user_id', 'bot_risk', 'events_per_sec']]
    bot_df['recommended_action'] = bot_df['bot_risk'].apply(lambda x: 'Block User' if x == 'High' else 'Flag & Monitor')
    if not bot_df.empty:
        print(bot_df.to_markdown(index=False))
    else:
        print("No significant bot behavior detected.")

def _print_recommendations():
    print("\n\n--- 5. Recommendations ---")
    print("""
    **App Stability:**
    1.  **Prioritize Patches:** Immediately investigate and fix crashes on older versions (2.0.3, 2.0.5, 2.0.8).
    2.  **Investigate Startup Crash:** The zero-activity crash (S019) is critical and likely prevents app launch.
    
    **Bot Prevention:**
    1.  **Block High-Risk Users:** Immediately block users flagged with 'High' bot risk, especially repeat offenders like U1002.
    2.  **Implement Rate Limiting:** Introduce server-side limits on event and screen view rates per second to prevent abuse.
    
    **Data Quality:**
    1.  **Enforce Mandatory Fields:** Make `device_id` and `session_end` non-nullable in the data logging schema.
    2.  **Monitor Anomalies:** Regularly review reports for long sessions and minimal activity sessions to identify potential usability issues.
    """)

def _generate_visuals(df, output_dir):
    print(f"\n[INFO] Generating visual charts and saving to '{output_dir}' directory...")
    
    # Crash Chart
    plt.figure(figsize=(10, 6))
    crashed_sessions = df[df['crash'] == True]
    sns.countplot(data=crashed_sessions, x='app_version', hue='platform', palette='viridis')
    plt.title('Crash Counts by App Version and Platform', fontsize=16)
    plt.savefig(os.path.join(output_dir, 'crash_summary_chart.png'))
    plt.close()
    
    # Bot Activity Chart
    plt.figure(figsize=(10, 6))
    df['is_bot_flagged'] = df['bot_risk'].isin(['Medium', 'High'])
    sns.scatterplot(data=df, x='duration_sec', y='events_count', hue='is_bot_flagged', palette={True: 'red', False: 'blue'}, alpha=0.7)
    plt.title('Bot-like Activity vs. Normal Sessions', fontsize=16)
    plt.xlabel('Session Duration (seconds)')
    plt.ylabel('Number of Events')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(os.path.join(output_dir, 'bot_activity_chart.png'))
    plt.close()
    
    print("[SUCCESS] Charts generated successfully.")