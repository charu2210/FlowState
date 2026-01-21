import joblib
import numpy as np
import os

class CognitiveLoadModel:
    def __init__(self, model_path="src/flowstate_model.pkl"):
        """
        Loads the trained Isolation Forest model.
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"❌ Model not found at {model_path}. Run train_model.py first.")
        
        self.model = joblib.load(model_path)
        self.window_size = 50  # Must match training window

    def _extract_features(self, flight_times):
        """
        Converts raw timings into the features the model expects:
        [Rolling Mean, Rolling Std Dev]
        """
        if len(flight_times) < self.window_size:
            return None
            
        # Use only the last N keystrokes
        window = flight_times[-self.window_size:]
        
        mean_val = np.mean(window)
        std_val = np.std(window)
        
        # Reshape for Scikit-Learn: [[mean, std]]
        return np.array([[mean_val, std_val]])

    def predict(self, flight_times):
        """
        Returns:
            1  = Normal State (Flow)
            -1 = Anomaly (Fatigue/High Cognitive Load)
            0  = Not enough data yet
        """
        features = self._extract_features(flight_times)
        
        if features is None:
            return 0
        
        # Isolation Forest returns [1] or [-1]
        prediction = self.model.predict(features)
        return prediction[0]