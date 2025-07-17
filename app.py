from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd

# Initialize FastAPI app
app = FastAPI()

# Enable CORS to allow requests from your frontend (adjust origin in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to ["https://yourdomain.com"] in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load trained ML model
model = joblib.load("model.joblib")

# Define expected features (must match training order)
expected_features = [
    "soil_type", "depth", "area", "equipment_type", "distance_to_dump",
    "operator_experience", "weather_condition", "groundwater_level", "rock_content"
]

# Helper: Convert incoming data to DataFrame
def preprocess_input(data: dict) -> pd.DataFrame:
    df = pd.DataFrame([data])
    return df[expected_features]  # Ensure column order

# API Route: Predict excavation time
@app.post("/predict")
async def predict(request: Request):
    try:
        data = await request.json()
        input_df = preprocess_input(data)
        prediction = model.predict(input_df)[0]
        return {
            "predicted_time_hours": round(prediction, 2)
        }
    except Exception as e:
        return {"error": str(e)}
