import os
import librosa
import numpy as np
import joblib

# Path to your dataset folder
DATA_FOLDER = "data"

# Maqams (must match your folder names inside /data)
MAQAMS = [
    "maqam_ajam", "maqam_bayati", "maqam_hijaz", "maqam_kurd",
    "maqam_nahawand", "maqam_rast", "maqam_saba", "maqam_sikah"
]

def extract_features(file_path):
    try:
        y, sr = librosa.load(file_path, duration=30)
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        return np.mean(mfcc.T, axis=0)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Prepare dataset
X, y = [], []
for maqam in MAQAMS:
    folder = os.path.join(DATA_FOLDER, maqam)
    if not os.path.exists(folder):
        print(f"Warning: folder {folder} not found, skipping.")
        continue
    for file_name in os.listdir(folder):
        if file_name.lower().endswith(".mp3") or file_name.lower().endswith(".wav"):
            file_path = os.path.join(folder, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(maqam)

X = np.array(X)
y = np.array(y)

# Train classifier
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# Save model
joblib.dump(clf, "model.pkl")
print("✅ Model trained and saved as model.pkl")
