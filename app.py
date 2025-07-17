from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# Load the trained model
model = joblib.load("model.joblib")

# Define the FastAPI app
app = FastAPI()

# Define input data model using Pydantic
class ExcavationInput(BaseModel):
    Soil_Type: str
    Excavation_Type: str
    Volume_m3: float
    Depth_m: float
    Equipment_Type: str
    Workers: int
    Weather: str
    Season: str

# Define prediction route
@app.post("/predict")
def predict(data: ExcavationInput):
    input_df = pd.DataFrame([data.dict()])
    prediction = model.predict(input_df)[0]
    return {"predicted_time_hr": round(prediction, 2)}
