from typing import Any, List, Optional

from pydantic import BaseModel


class Health(BaseModel):
    name: str
    api_version: str
    model_version: str


class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class TitanicDataInputSchema(BaseModel):
    pclass: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[int]
    sibsp: Optional[str]
    parch: Optional[int]
    ticket: Optional[str]
    fare: Optional[float]
    cabin: Optional[str]
    embarked: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "pclass": 1,
                        "name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                        "sex": "female",
                        "age": 38,
                        "sibsp": "1",
                        "parch": 0,
                        "ticket": "PC 17599",
                        "fare": 71.2833,
                        "cabin": "C85",
                        "embarked": "C",
                    }
                ]
            }
        }
