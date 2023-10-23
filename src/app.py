from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import pandas as pd

# Instantiate The Fast API instance
app = FastAPI(
    title="Sepsis Prediction API",
    description="This FastAPI application provides sepsis predictions using a machine learning model.",
    version="1.0"
)

# Load the model and key components
with open('model_and_key_components.pkl', 'rb') as file:
    loaded_components = pickle.load(file)

loaded_model = loaded_components['model']
loaded_encoder = loaded_components['encoder']
loaded_scaler = loaded_components['scaler']

# Define the input data structure using Pydantic BaseModel
class InputData(BaseModel):
    PRG: int
    PL: float
    PR: float
    SK: float
    TS: int
    M11: float
    BD2: float
    Age: int

# Define the output data structure using Pydantic BaseModel
class OutputData(BaseModel):
    Sepsis: str

# Define a function to preprocess input data
def preprocess_input_data(input_data: InputData):
    # Encode Categorical Variables (if needed)
    # All columns are numerical. No need for encoding

    # Apply scaling to numerical data
    numerical_cols = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']
    input_data_scaled = loaded_scaler.transform([list(input_data.dict().values())])

    return pd.DataFrame(input_data_scaled, columns=numerical_cols)

# Define a function to make predictions
def make_predictions(input_data_scaled_df: pd.DataFrame):
    y_pred = loaded_model.predict(input_data_scaled_df)
    sepsis_mapping = {0: 'Negative', 1: 'Positive'}
    return sepsis_mapping[y_pred[0]]

@app.get("/")
async def root():
    # Endpoint at the root URL ("/") returns a welcome message with a clickable link
    message = "Welcome to your Sepsis Classification API! Click [here](/docs) to access the API documentation."
    return {"message": message}


@app.post("/predict/", response_model=OutputData)
async def predict_sepsis(input_data: InputData):
    try:
        input_data_scaled_df = preprocess_input_data(input_data)
        sepsis_status = make_predictions(input_data_scaled_df)
        return {"Sepsis": sepsis_status}
    except Exception as e:

        # Handle exceptions and return an error response
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Run the FastAPI application on the local host and port 7860
    CMD ["uvicorn", "app:app", "--host", "127.0.0.1", "--port", "7860", "--reload"]