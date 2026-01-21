import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

def generate_plot():
    # 1. Load Data
    # We look for data in the data/raw_logs folder relative to this script
    data_path = os.path.join("data", "raw_logs")
    all_files = glob.glob(os.path.join(data_path, "*.csv"))

    if not all_files:
        print("❌ No data found! Please run logger.py first.")
        return

    print(f"Loading {len(all_files)} log files for visualization...")
    df_list = [pd.read_csv(f) for f in all_files]
    df = pd.concat(df_list, ignore_index=True)

    # 2. Feature Engineering (The "Science")
    # Rolling Standard Deviation = Rhythm Consistency
    df['rolling_std'] = df['flight_time'].rolling(window=50).std()
    df['rolling_mean'] = df['flight_time'].rolling(window=50).mean()

    # 3. Plotting
    plt.figure(figsize=(12, 6))

    # Plot Raw Data (Gray dots)
    plt.plot(df.index, df['flight_time'], 'o', markersize=2, alpha=0.2, color='gray', label='Raw Keystrokes')

    # Plot The "Flow" (Blue Line)
    plt.plot(df.index, df['rolling_mean'], linewidth=2, color='#2563eb', label='Typing Speed (Avg)')

    # Plot The "Fatigue" (Red Line) - Scaled up to be visible
    plt.plot(df.index, df['rolling_std'] * 4, linewidth=2, color='#dc2626', label='Variance (Fatigue Indicator)')

    plt.title(f"FlowState Analysis: {len(df)} Keystrokes", fontsize=14)
    plt.xlabel("Keystroke Count")
    plt.ylabel("Flight Time (seconds)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save to the ROOT folder so README can find it
    output_path = "dynamics_plot.png"
    plt.savefig(output_path, dpi=300)
    print(f"✅ Graph saved to: {output_path}")
    print("Check your project folder!")

if __name__ == "__main__":
    generate_plot()