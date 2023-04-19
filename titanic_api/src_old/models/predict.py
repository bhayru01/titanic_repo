from typing import Union

import pandas as pd

from src.config.core import config
from src import __version__ as _version
from src.processing.data_manager import load_pipeline
from src.processing.validation import validate_inputs

# load model
pipeline_file_name = f"{config.app_config.pipeline_save_file}.pkl"
_price_pipe = load_pipeline(pipeline_file_name)


def make_prediction(input_data: Union[pd.DataFrame, dict]) -> dict:
    """Make a prediction using a saved model pipeline."""

    data = pd.DataFrame(input_data)

    # validate data
    validated_data, errors = validate_inputs(data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _price_pipe.predict(validated_data[config.model_config.features])
        results["predictions"] = list(predictions)
        results["version"] = _version
        results["errors"] = errors

    return results
