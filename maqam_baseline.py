# maqaam_baseline.py
import os, glob
import numpy as np
import librosa
import soundfile as sf
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import LeaveOneOut
from sklearn.metrics import classification_report, confusion_matrix

DATA_DIR = "data"  # folder with subfolders per maqam

def extract_features(y, sr):
    # 1) MFCCs
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = mfcc.mean(axis=1)
    mfcc_var = mfcc.var(axis=1)
    # 2) Chroma
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    chroma_mean = chroma.mean(axis=1)
    # 3) Pitch contour using pyin (voiced)
    try:
        f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'), fmax=librosa.note_to_hz('C7'))
        f0_mean = np.nanmean(f0)
        f0_std = np.nanstd(f0[np.isfinite(f0)]) if np.any(np.isfinite(f0)) else 0.0
    except Exception:
        f0_mean, f0_std = 0.0, 0.0
    # Aggregate
    feat = np.concatenate([mfcc_mean, mfcc_var, chroma_mean, [f0_mean, f0_std]])
    return feat

X, y = [], []
labels = sorted(os.listdir(DATA_DIR))
for lab in labels:
    folder = os.path.join(DATA_DIR, lab)
    if not os.path.isdir(folder): continue
    for fn in glob.glob(os.path.join(folder, "*.wav")):
        y_audio, sr = librosa.load(fn, sr=22050, mono=True)
        feat = extract_features(y_audio, sr)
        X.append(feat)
        y.append(lab)

X = np.vstack(X)
y = np.array(y)

# Handle varying feature length or NaNs
X = np.nan_to_num(X)

# Cross-validate using leave-one-out (good for small datasets)
loo = LeaveOneOut()
y_true, y_pred = [], []
clf = RandomForestClassifier(n_estimators=200, random_state=42)

for train_idx, test_idx in loo.split(X):
    clf.fit(X[train_idx], y[train_idx])
    p = clf.predict(X[test_idx])
    y_true.append(y[test_idx][0])
    y_pred.append(p[0])

print(classification_report(y_true, y_pred, digits=4))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred, labels=labels))
