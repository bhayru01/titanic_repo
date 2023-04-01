from src.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
import pandas as pd
from sklearn.pipeline import Pipeline
import joblib
from typing import Tuple


def load_dataset(data_url: str) -> pd.DataFrame:
    data = pd.read_csv(data_url)
    return data

def sequential_split(data: pd.DataFrame,
                     split_ratio: float,
                     target: str) -> Tuple[pd.DataFrame]:
    train_test_split_point = round(len(data) * (1 - split_ratio))
    train = data.iloc[:train_test_split_point]
    test = data.iloc[train_test_split_point::].drop(target, axis=1)
    y_test = data.iloc[train_test_split_point::][target]
    return train, test, y_test

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