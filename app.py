from fastapi import FastAPI, File, UploadFile
import uvicorn
import librosa
import numpy as np
import pickle

app = FastAPI()

# Load model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050, mono=True)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    features = np.concatenate([mfcc.mean(axis=1), chroma.mean(axis=1)])
    return features.reshape(1, -1)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    file_path = f"temp_{file.filename}"
    with open(file_path, "wb") as f:
        f.write(await file.read())
    features = extract_features(file_path)
    pred = model.predict(features)[0]
    return {"maqam": pred}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
