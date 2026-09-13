import pickle
from pathlib import Path

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field

from starter.starter.ml.data import process_data
from starter.starter.ml.model import inference, train_model

MODEL_PATH = Path(__file__).resolve().parent / "model" / "census_model.pkl"
DATA_PATH = Path(__file__).resolve().parent / "data" / "census.csv"

app = FastAPI(title="Census Salary Predictor")


class CensusInput(BaseModel):
    age: int
    workclass: str
    fnlgt: int
    education: str
    education_num: int = Field(alias="education-num")
    marital_status: str = Field(alias="marital-status")
    occupation: str
    relationship: str
    race: str
    sex: str
    capital_gain: int = Field(alias="capital-gain")
    capital_loss: int = Field(alias="capital-loss")
    hours_per_week: int = Field(alias="hours-per-week")
    native_country: str = Field(alias="native-country")

    model_config = {
        "populate_by_name": True,
        "json_schema_extra": {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States",
            }
        },
    }


class PredictionResponse(BaseModel):
    prediction: str


def _load_model_artifacts():
    if not MODEL_PATH.exists():
        train_model_file()

    with MODEL_PATH.open("rb") as f:
        return pickle.load(f)


def train_model_file():
    data = pd.read_csv(DATA_PATH)
    data.columns = [c.strip() for c in data.columns]

    categorical_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, y, encoder, lb = process_data(
        data,
        categorical_features=categorical_features,
        label="salary",
        training=True,
    )
    model = train_model(X, y)

    with MODEL_PATH.open("wb") as f:
        pickle.dump({"model": model, "encoder": encoder, "lb": lb}, f)

    return model


@app.get("/")
def read_root():
    return {"message": "Welcome to the census income prediction API."}


@app.post("/predict", response_model=PredictionResponse)
def predict(payload: CensusInput):
    artifacts = _load_model_artifacts()
    model = artifacts["model"]
    encoder = artifacts["encoder"]
    lb = artifacts["lb"]

    input_df = pd.DataFrame(
        [
            {
                "age": payload.age,
                "workclass": payload.workclass,
                "fnlgt": payload.fnlgt,
                "education": payload.education,
                "education-num": payload.education_num,
                "marital-status": payload.marital_status,
                "occupation": payload.occupation,
                "relationship": payload.relationship,
                "race": payload.race,
                "sex": payload.sex,
                "capital-gain": payload.capital_gain,
                "capital-loss": payload.capital_loss,
                "hours-per-week": payload.hours_per_week,
                "native-country": payload.native_country,
            }
        ]
    )

    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    X, _, _, _ = process_data(
        input_df,
        categorical_features=cat_features,
        label=None,
        training=False,
        encoder=encoder,
        lb=lb,
    )

    prediction = inference(model, X)[0]
    return {"prediction": "<=50K" if prediction == 0 else ">50K"}
