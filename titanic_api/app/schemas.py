from pydantic import BaseModel
from typing import Any, Optional, List

class Health(BaseModel):
    name: str
    api_version: str
    model_version: str

class PredictionResults(BaseModel):
    errors: Optional[Any]
    version: str
    predictions: Optional[List[float]]


class TitanicDataInputSchema(BaseModel):
    Pclass: Optional[int]
    Name: Optional[str]
    Sex: Optional[str]
    Age: Optional[int]
    SibSp: Optional[str]
    Parch: Optional[int]
    Ticket: Optional[str]
    Fare: Optional[float]
    Cabin: Optional[str]
    Embarked: Optional[str]


class MultipleTitanicDataInputs(BaseModel):
    inputs: List[TitanicDataInputSchema]

    class Config:
        schema_extra = {
            "example": {
                "inputs": [
                    {
                        "Pclass": 1,
                        "Name": "Cumings, Mrs. John Bradley (Florence Briggs Thayer)",
                        "Sex": "female",
                        "Age": 38,
                        "SibSp": "1",
                        "Parch": 0,
                        "Ticket": "PC 17599",
                        "Fare": 71.2833,
                        "Cabin": "C85",
                        "Embarked": "C"
                    }
                ]
            }
        }