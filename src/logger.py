import csv
import time
import os
from pynput import keyboard

# CONFIGURATION
DATA_DIR = "data/raw_logs"
os.makedirs(DATA_DIR, exist_ok=True)
LOG_FILE = os.path.join(DATA_DIR, f"session_{int(time.time())}.csv")

# BUFFER
key_data = []
last_release_time = None

def init_csv():
    """Creates the CSV file with headers if it doesn't exist."""
    with open(LOG_FILE, 'w', newline='') as f:
        writer = csv.writer(f)
        # We record:
        # timestamp: When it happened
        # dwell_time: How long key was down (placeholder for now)
        # flight_time: Time since last key release
        writer.writerow(["timestamp", "flight_time"])

def on_press(key):
    global last_release_time
    press_time = time.time()
    
    # Calculate Flight Time (Time since PREVIOUS key release)
    if last_release_time is not None:
        flight_time = press_time - last_release_time
        
        # Filter: Ignore pauses longer than 2 seconds (thinking time)
        if flight_time < 2.0:
            save_data(press_time, flight_time)

def on_release(key):
    global last_release_time
    last_release_time = time.time()
    
    # Stop loop if ESC is pressed (Optional)
    if key == keyboard.Key.esc:
        return False

def save_data(timestamp, flight_time):
    """Appends a single keystroke metric to the CSV."""
    with open(LOG_FILE, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, flight_time])

if __name__ == "__main__":
    print(f"✅ Logging started. Data saving to {LOG_FILE}")
    print("🔒 Privacy Mode: ON (Key values are NOT recorded)")
    print("Press ESC to stop.")
    
    init_csv()
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()