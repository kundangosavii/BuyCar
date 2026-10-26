
import json
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sklearn.compose import _column_transformer

from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

BASE_DIR = Path(__file__).resolve().parent


def _load_model():
    """Load the model across the scikit-learn versions used to create it."""
    # scikit-learn 1.6 serialized this private list subclass, which was
    # removed in later releases. Its list behavior is all the saved model
    # needs during unpickling and prediction.
    if not hasattr(_column_transformer, "_RemainderColsList"):
        class _RemainderColsList(list):
            pass

        _column_transformer._RemainderColsList = _RemainderColsList

    loaded_model = joblib.load(BASE_DIR / "car_price_model.joblib")

    # SimpleImputer renamed this fitted dtype attribute after the model was
    # saved. Restore the attribute expected by the current implementation.
    visited = set()

    def restore_attributes(value):
        if id(value) in visited:
            return
        visited.add(id(value))

        if value.__class__.__name__ == "SimpleImputer":
            if not hasattr(value, "_fill_dtype") and hasattr(value, "_fit_dtype"):
                value._fill_dtype = value._fit_dtype

        if isinstance(value, dict):
            for item in value.values():
                restore_attributes(item)
        elif isinstance(value, (list, tuple, set)):
            for item in value:
                restore_attributes(item)
        elif hasattr(value, "__dict__"):
            for item in vars(value).values():
                restore_attributes(item)

    restore_attributes(loaded_model)
    return loaded_model


# Load the saved model and metadata once at startup
model = _load_model()

with open(BASE_DIR / "car_metadata.json", "r", encoding="utf-8") as f:
    metadata = json.load(f)

app = FastAPI(
    title="Used Car Price Predictor",
    description="Predict a used car's selling price using machine learning.",
    version="1.0.0",
)


class CarDetails(BaseModel):
    name: str
    year: int = Field(ge=1900, le=2100)
    km_driven: int = Field(ge=0)
    fuel: str
    seller_type: str
    transmission: str
    owner: str

STATIC_DIR = BASE_DIR / "static"
STATIC_DIR.mkdir(exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", include_in_schema=False)
def serve_frontend():
    return FileResponse(STATIC_DIR / "index.html")

@app.get("/")
def home():
    return {
        "message": "Used Car Price Predictor API is running",
        "docs": "/docs",
    }


@app.get("/options")
def get_options():
    return metadata


@app.post("/predict")
def predict_price(car: CarDetails):
    car_data = pd.DataFrame([car.model_dump()])

    try:
        predicted_price = float(model.predict(car_data)[0])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail="Prediction failed. Check model compatibility and input features.",
        ) from exc

    return {
        "predicted_price": round(max(0.0, predicted_price), 2),
        "currency": "INR",
        "note": "This is a model estimate, not a guaranteed market price.",
    }
