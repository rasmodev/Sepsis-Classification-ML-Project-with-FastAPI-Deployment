# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import numpy as np

# Load the model and components
loaded_components = joblib.load('model_and_key_components.pkl')
loaded_model = loaded_components['model']
loaded_scaler = loaded_components['scaler']

# Create an instance of the FastAPI class
app = FastAPI()

# Create a Pydantic model for input data
class InputData(BaseModel):
    PRG: int
    PL: float
    PR: float
    SK: float
    TS: int
    M11: float
    BD2: float
    Age: int

# Define a route for making predictions
@app.post("/predict/")
async def predict(data: InputData):
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Apply scaling to numerical data
        numerical_cols = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']
        input_data_scaled = loaded_scaler.transform(input_data[numerical_cols])

        # Make predictions using the loaded model
        prediction = loaded_model.predict(input_data_scaled)

        # Map the prediction to 'Negative' or 'Positive'
        sepsis_mapping = {0: 'Negative', 1: 'Positive'}
        prediction_label = sepsis_mapping[prediction[0]]

        return {"prediction": prediction_label}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI application with Uvicorn
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
