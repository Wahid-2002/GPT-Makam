import os
import pickle
import joblib
from fastapi import FastAPI

app = FastAPI()

MODEL_PATH = "model.pkl"

def load_model(path):
    # Try loading with pickle first
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except pickle.UnpicklingError:
        # If pickle fails, try joblib
        return joblib.load(path)

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

model = load_model(MODEL_PATH)

@app.get("/")
def home():
    return {"message": "Model loaded successfully!", "model_type": str(type(model))}
