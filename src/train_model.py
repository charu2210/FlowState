import pandas as pd
import numpy as np
import glob
import os
import joblib
from sklearn.ensemble import IsolationForest

# CONFIGURATION
DATA_DIR = "data/raw_logs"
MODEL_PATH = "src/flowstate_model.pkl"

def load_and_process_data():
    """
    1. Loads all CSV files from data/raw_logs.
    2. Calculates Rolling Statistics (Feature Engineering).
    """
    all_files = glob.glob(os.path.join(DATA_DIR, "*.csv"))
    
    if not all_files:
        raise ValueError("No data found! Run the logger.py script first to collect keystrokes.")

    print(f"Loading {len(all_files)} log files...")
    df_list = []
    
    for filename in all_files:
        df = pd.read_csv(filename)
        df_list.append(df)
    
    full_df = pd.concat(df_list, ignore_index=True)
    
    # --- RESEARCH FEATURE ENGINEERING ---
    # We don't care about single keys. We care about the "Rhythm" over time.
    # Window size 50 = We look at the last 50 keystrokes to judge current state.
    
    # 1. Rolling Mean (Speed)
    full_df['rolling_mean'] = full_df['flight_time'].rolling(window=50).mean()
    
    # 2. Rolling Std Dev (Consistency) - The key indicator of fatigue
    full_df['rolling_std'] = full_df['flight_time'].rolling(window=50).std()
    
    # Drop NaN values created by the rolling window
    clean_df = full_df.dropna()[['rolling_mean', 'rolling_std']]
    
    return clean_df

def train_isolation_forest(df):
    """
    Trains an Isolation Forest to learn 'Normal' behavior.
    """
    print(f"Training on {len(df)} samples...")
    
    # contamination=0.05 means we assume 5% of your data represents 'Fatigue/Distraction'
    model = IsolationForest(n_estimators=100, contamination=0.05, random_state=42)
    model.fit(df)
    
    return model

if __name__ == "__main__":
    try:
        # 1. Prepare Data
        data = load_and_process_data()
        
        # 2. Train Model
        model = train_isolation_forest(data)
        
        # 3. Save Model
        joblib.dump(model, MODEL_PATH)
        print(f"✅ Model saved successfully to {MODEL_PATH}")
        print("You can now run 'src/live_monitor.py'")
        
    except Exception as e:
        print(f"❌ Error: {e}")