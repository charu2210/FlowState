import time
from pynput import keyboard
from inference import CognitiveLoadModel # Importing your new class

# LOAD THE BRAIN
try:
    brain = CognitiveLoadModel()
    print("✅ AI Model Loaded Successfully.")
except Exception as e:
    print(e)
    exit()

# LIVE BUFFER
flight_times = []
last_release_time = None

def trigger_intervention():
    """
    The 'Action' layer. 
    """
    print("\n" + "!"*50)
    print("⚠️  COGNITIVE LOAD HIGH DETECTED - TAKE A BREAK ⚠️")
    print("!"*50 + "\n")

def on_press(key):
    global last_release_time
    press_time = time.time()
    
    if last_release_time is not None:
        flight_time = press_time - last_release_time
        
        # Filter pauses (thinking time)
        if flight_time < 2.0: 
            flight_times.append(flight_time)
            
            # ASK THE BRAIN
            # We just pass the raw data; the class handles the math.
            status = brain.predict(flight_times)
            
            if status == -1:
                trigger_intervention()
                # Optional: Clear buffer to prevent spamming
                # flight_times = [] 
            elif status == 0:
                pass # Not enough data yet
            else:
                print(".", end="", flush=True) # Normal flow

            # Keep buffer manageable size
            if len(flight_times) > 100:
                flight_times.pop(0)

def on_release(key):
    global last_release_time
    last_release_time = time.time()
    if key == keyboard.Key.esc:
        return False

if __name__ == "__main__":
    print("✅ FlowState Monitor Running...")
    print("Typing normally... (Alerts will trigger on fatigue)")
    
    with keyboard.Listener(on_press=on_press, on_release=on_release) as listener:
        listener.join()