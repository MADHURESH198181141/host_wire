import pandas as pd

def create_full_report(df):
    """Creates a full analysis report."""
    print("=== App Usage Analysis Report ===")
    print(f"Total sessions: {len(df)}")
    print(f"Crashed sessions: {df['crash'].sum()}")
    print(f"Bot risk levels: {df['bot_risk'].value_counts().to_dict()}")
    # Save detailed report
    df.to_csv('reports/full_analysis.csv', index=False)
    print("Detailed report saved to reports/full_analysis.csv")
