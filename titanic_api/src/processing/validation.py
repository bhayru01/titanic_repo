import pandas as pd
import numpy as np
from typing import List, Optional, Tuple
from src.config.core import config
from pydantic import BaseModel, ValidationError
from src.processing.preprocessors import DropVars, ReplaceQuestionMarks, CastVarsToFloat
from sklearn.pipeline import Pipeline


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

def drop_na_inputs(input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validate_data = input_data.copy()
    new_vars_with_na = [
        var for var in validate_data
        if var
        not in config.model_config.categorical_vars
        + config.model_config.numerical_vars
        and validate_data[var].isnull().sum() > 0
    ]
    validate_data.dropna(subset = new_vars_with_na, inplace = True)

    return validate_data


def validate_inputs(input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""
    relevant_data = input_data[config.model_config.features].copy()
    validated_data = drop_na_inputs(relevant_data)

    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        inputs = validated_data.replace({np.nan: None}).to_dict(orient="records")
        MultipleTitanicDataInputs(inputs=inputs)
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors




