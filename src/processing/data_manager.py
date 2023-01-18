from src.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib


def load_dataset(filename: str) -> pd.DataFrame:
    df = pd.read_csv(DATASET_DIR / filename)
    return df


def save_pipeline(pipeline: Pipeline) -> None:
    """Save pipeline """

    file_name = f'{config.app_config.pipeline_save_file}.pkl'
    save_path = TRAINED_MODEL_DIR / file_name
    joblib.dump(pipeline, save_path)


def load_pipeline(file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(file_path)
    return trained_model