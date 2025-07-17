from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

app = FastAPI()

# Enable CORS (allow all origins for testing; restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://your-frontend.com"] for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model
model = joblib.load("model.joblib")

# Define input format
class ExcavationInput(BaseModel):
    Soil_Type: str
    Excavation_Type: str
    Volume_m3: float
    Depth_m: float
    Equipment_Type: str
    Workers: int
    Weather: str
    Season: str

@app.get("/")
def root():
    return {"message": "Excavation Time Predictor is Live!"}

@app.post("/predict")
def predict(data: ExcavationInput):
    input_df = pd.DataFrame([data.dict()])
    pred = model.predict(input_df)[0]
    return {"predicted_time_hr": round(pred, 2)}
